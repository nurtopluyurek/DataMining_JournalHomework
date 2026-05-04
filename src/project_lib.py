from __future__ import annotations

import html
import hashlib
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, silhouette_score, top_k_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import LinearSVC


DEFAULT_SQLITE_PATH = Path(r"C:\Users\toplu\Downloads\CompSciencePub (1).sqlite")
DEFAULT_RANDOM_STATE = 42
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = DEFAULT_PROJECT_ROOT / "outputs" / "cache"
DEFAULT_WORKSPACE_SQLITE_PATH = DEFAULT_PROJECT_ROOT / "data" / "CompSciencePub (1).sqlite"
DEFAULT_PREPARED_DATASET_PATH = DEFAULT_CACHE_DIR / "prepared_dataset.pkl"
DEFAULT_SBERT_MODEL_NAME = "all-MiniLM-L6-v2"


def strip_html(text: Any) -> str:
    if text is None or pd.isna(text):
        return ""
    value = html.unescape(str(text))
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def clean_text(text: Any) -> str:
    value = strip_html(text).lower()
    value = re.sub(r"[^a-z0-9\-\+\./ ]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def split_aggregate_field(text: Any) -> list[str]:
    raw_value = strip_html(text)
    if not raw_value:
        return []
    parts = [part.strip() for part in raw_value.split("|")]
    deduped: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if not part:
            continue
        key = part.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(part)
    return deduped


def _join_display(values: Iterable[str]) -> str:
    return "; ".join(value for value in values if value)


def _join_clean(values: Iterable[str]) -> str:
    return " ".join(clean_text(value) for value in values if clean_text(value)).strip()


DATASET_QUERY = """
WITH keyword_map AS (
    SELECT
        ark.AcademicRecordId,
        GROUP_CONCAT(ak.Name, ' | ') AS keywords_raw
    FROM AcademicRecordKeyword ark
    JOIN AcademicKeyword ak
        ON ak.AcademicKeywordID = ark.AcademicKeywordId
    GROUP BY ark.AcademicRecordId
),
keyword_plus_map AS (
    SELECT
        arkp.AcademicRecordId,
        GROUP_CONCAT(akp.Name, ' | ') AS keyword_plus_raw
    FROM AcademicRecordKeywordPlus arkp
    JOIN AcademicKeywordPlus akp
        ON akp.AcademicKeywordPlusID = arkp.AcademicKeywordPlusId
    GROUP BY arkp.AcademicRecordId
),
subject_map AS (
    SELECT
        ars.AcademicRecordId,
        GROUP_CONCAT(s.NameEn, ' | ') AS subjects_raw
    FROM AcademicRecordSubject ars
    JOIN AcademicSubject s
        ON s.AcademicSubjectID = ars.AcademicSubjectId
    GROUP BY ars.AcademicRecordId
)
SELECT
    ar.AcademicRecordID AS record_id,
    ar.WosUID AS wos_uid,
    ar.Title AS title_raw,
    ara.AbstractText AS abstract_raw,
    p.Name AS journal,
    ar.PubYear AS pub_year,
    dt.NameEn AS document_type,
    pt.NameEn AS publication_type,
    sm.subjects_raw,
    km.keywords_raw,
    kpm.keyword_plus_raw
FROM AcademicRecord ar
LEFT JOIN AcademicRecordAbstract ara
    ON ara.AcademicRecordId = ar.AcademicRecordID
LEFT JOIN Publication p
    ON p.PublicationID = ar.PublicationId
LEFT JOIN DocumentType dt
    ON dt.DocumentTypeID = ar.DocumentTypeId
LEFT JOIN PublicationType pt
    ON pt.PublicationTypeID = ar.PublicationTypeId
LEFT JOIN subject_map sm
    ON sm.AcademicRecordId = ar.AcademicRecordID
LEFT JOIN keyword_map km
    ON km.AcademicRecordId = ar.AcademicRecordID
LEFT JOIN keyword_plus_map kpm
    ON kpm.AcademicRecordId = ar.AcademicRecordID
"""


def load_dataset(sqlite_path: str | Path = DEFAULT_SQLITE_PATH) -> pd.DataFrame:
    requested_path = Path(sqlite_path)
    candidate_paths: list[Path] = [requested_path]
    if requested_path != DEFAULT_SQLITE_PATH:
        candidate_paths.append(DEFAULT_SQLITE_PATH)
    if DEFAULT_WORKSPACE_SQLITE_PATH not in candidate_paths:
        candidate_paths.append(DEFAULT_WORKSPACE_SQLITE_PATH)

    frame = None
    last_error: Exception | None = None
    for path in candidate_paths:
        try:
            with sqlite3.connect(path) as connection:
                frame = pd.read_sql_query(DATASET_QUERY, connection)
            break
        except Exception as exc:  # pragma: no cover - exercised only in environment-specific fallback paths
            last_error = exc
            continue

    if frame is None:
        if DEFAULT_PREPARED_DATASET_PATH.exists():
            return pd.read_pickle(DEFAULT_PREPARED_DATASET_PATH)
        attempted = ", ".join(str(path) for path in candidate_paths)
        raise RuntimeError(f"Could not load SQLite dataset from any known path: {attempted}") from last_error

    frame["title"] = frame["title_raw"].map(strip_html)
    frame["abstract"] = frame["abstract_raw"].map(strip_html)
    frame["subject_list"] = frame["subjects_raw"].map(split_aggregate_field)
    frame["keyword_list"] = frame["keywords_raw"].map(split_aggregate_field)
    frame["keyword_plus_list"] = frame["keyword_plus_raw"].map(split_aggregate_field)
    frame["subjects"] = frame["subject_list"].map(_join_display)
    frame["keywords"] = frame["keyword_list"].map(_join_display)
    frame["keyword_plus"] = frame["keyword_plus_list"].map(_join_display)

    frame["title_clean"] = frame["title"].map(clean_text)
    frame["abstract_clean"] = frame["abstract"].map(clean_text)
    frame["subjects_clean"] = frame["subject_list"].map(_join_clean)
    frame["keywords_clean"] = frame["keyword_list"].map(_join_clean)
    frame["keyword_plus_clean"] = frame["keyword_plus_list"].map(_join_clean)
    frame["title_abstract_text"] = (
        frame["title_clean"].fillna("").str.strip() + ". " + frame["abstract_clean"].fillna("").str.strip()
    ).str.strip(". ").str.strip()
    frame["abstract_only_text"] = frame["abstract_clean"].fillna("").str.strip()
    frame["abstract_keywords_text"] = (
        frame["abstract_clean"].fillna("")
        + ". keywords "
        + frame["keywords_clean"].fillna("")
    ).str.strip(". ").str.strip()
    frame["abstract_keywords_subjects_text"] = (
        frame["abstract_clean"].fillna("")
        + ". keywords "
        + frame["keywords_clean"].fillna("")
        + ". subjects "
        + frame["subjects_clean"].fillna("")
    ).str.strip(". ").str.strip()
    frame["combined_text"] = (
        frame["title_clean"].fillna("")
        + ". "
        + frame["abstract_clean"].fillna("")
        + ". keywords "
        + frame["keywords_clean"].fillna("")
        + ". keyword plus "
        + frame["keyword_plus_clean"].fillna("")
    ).str.strip(". ").str.strip()
    frame["abstract_query_text"] = frame["abstract_clean"]
    frame["journal_article_count"] = frame.groupby("journal")["journal"].transform("size")

    selected_columns = [
        "record_id",
        "wos_uid",
        "title",
        "abstract",
        "journal",
        "pub_year",
        "subjects",
        "keywords",
        "keyword_plus",
        "abstract_only_text",
        "abstract_keywords_text",
        "abstract_keywords_subjects_text",
        "combined_text",
        "title_abstract_text",
        "abstract_query_text",
        "document_type",
        "publication_type",
        "journal_article_count",
        "subject_list",
        "keyword_list",
        "keyword_plus_list",
    ]
    prepared_frame = frame[selected_columns].copy()
    try:  # pragma: no cover - cache write is environment-specific
        DEFAULT_PREPARED_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        prepared_frame.to_pickle(DEFAULT_PREPARED_DATASET_PATH)
    except Exception:
        pass
    return prepared_frame


def summarize_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    summary = {
        "total_records": int(len(frame)),
        "records_with_abstract": int(frame["abstract"].str.len().gt(0).sum()),
        "missing_abstracts": int(frame["abstract"].str.len().eq(0).sum()),
        "unique_journals": int(frame["journal"].nunique()),
        "document_types": int(frame["document_type"].nunique(dropna=True)),
        "publication_types": int(frame["publication_type"].nunique(dropna=True)),
        "min_pub_year": int(frame["pub_year"].dropna().min()),
        "max_pub_year": int(frame["pub_year"].dropna().max()),
        "avg_abstract_length": round(frame["abstract"].str.len().replace(0, np.nan).mean(), 2),
    }
    return pd.DataFrame([summary])


def filter_modeling_dataset(
    frame: pd.DataFrame,
    *,
    min_journal_frequency: int = 5,
    document_type: str = "Article",
) -> tuple[pd.DataFrame, dict[str, int]]:
    base_mask = frame["document_type"].fillna("").eq(document_type) & frame["abstract"].str.len().gt(0)
    filtered = frame.loc[base_mask].copy()
    journal_counts = filtered["journal"].value_counts()
    keep_journals = journal_counts[journal_counts >= min_journal_frequency].index
    long_tail_journals = journal_counts[journal_counts < min_journal_frequency].index
    model_frame = filtered.loc[filtered["journal"].isin(keep_journals)].copy().reset_index(drop=True)
    stats = {
        "starting_records": int(len(frame)),
        "article_records_with_abstract": int(len(filtered)),
        "model_records": int(len(model_frame)),
        "model_journals": int(model_frame["journal"].nunique()),
        "excluded_long_tail_journals": int(len(long_tail_journals)),
        "excluded_long_tail_records": int(filtered["journal"].isin(long_tail_journals).sum()),
        "min_examples_per_journal": int(min_journal_frequency),
    }
    return model_frame, stats


def train_test_split_by_journal(
    frame: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_index, test_index = train_test_split(
        frame.index.to_numpy(),
        test_size=test_size,
        random_state=random_state,
        stratify=frame["journal"],
    )
    train_frame = frame.loc[train_index].reset_index(drop=True)
    test_frame = frame.loc[test_index].reset_index(drop=True)
    return train_frame, test_frame


@dataclass
class EvaluationResult:
    name: str
    metrics: dict[str, float]
    predictions: pd.DataFrame
    confusion_pairs: pd.DataFrame


def _shorten_snippet(text: Any, max_chars: int = 180) -> str:
    snippet = strip_html(text)
    if len(snippet) <= max_chars:
        return snippet
    return snippet[: max_chars - 3].rstrip() + "..."


def _normalize_score_rows(score_matrix: np.ndarray) -> np.ndarray:
    score_matrix = np.asarray(score_matrix, dtype=np.float64)
    min_values = score_matrix.min(axis=1, keepdims=True)
    max_values = score_matrix.max(axis=1, keepdims=True)
    normalized = (score_matrix - min_values) / np.maximum(max_values - min_values, 1e-9)
    row_sums = normalized.sum(axis=1, keepdims=True)
    return normalized / np.maximum(row_sums, 1e-9)


def _pick_cluster_label_from_subjects(subject_names: Iterable[str], top_terms: Iterable[str]) -> str:
    for subject_name in subject_names:
        cleaned = subject_name.strip()
        if not cleaned or cleaned.casefold() == "computer science":
            continue
        if cleaned.casefold().startswith("computer science, "):
            return cleaned.split(",", 1)[1].strip()
        return cleaned
    term_list = [term.strip() for term in top_terms if term.strip()]
    if term_list:
        return term_list[0].replace("-", " ").title()
    return "General Computer Science"


def attach_cluster_annotations(frame: pd.DataFrame, clusterer: "TopicClusterer") -> pd.DataFrame:
    if len(frame) != len(clusterer.frame_):
        raise ValueError("Cluster annotations require the same row ordering and length as the fitted clusterer frame.")

    cluster_summary = clusterer.summarize_clusters().copy()
    cluster_label_map: dict[int, str] = {}
    cluster_description_map: dict[int, str] = {}
    for _, row in cluster_summary.iterrows():
        subject_names = [item.strip() for item in str(row["dominant_subjects"]).split(",") if item.strip()]
        top_terms = [item.strip() for item in str(row["top_terms"]).split(",") if item.strip()]
        cluster_id = int(row["cluster"])
        cluster_label_map[cluster_id] = _pick_cluster_label_from_subjects(subject_names, top_terms)
        cluster_description_map[cluster_id] = f"Subjects: {row['dominant_subjects']}. Terms: {row['top_terms']}."

    annotated = frame.copy()
    annotated["cluster_id"] = clusterer.labels_
    annotated["cluster_label"] = annotated["cluster_id"].map(cluster_label_map).fillna("General Computer Science")
    annotated["cluster_description"] = annotated["cluster_id"].map(cluster_description_map).fillna("")
    return annotated


def metrics_row(name: str, metrics: dict[str, float]) -> dict[str, float | str]:
    return {
        "model": name,
        "top_1_accuracy": metrics["top_1_accuracy"],
        "top_3_accuracy": metrics["top_3_accuracy"],
        "top_5_accuracy": metrics["top_5_accuracy"],
        "macro_f1": metrics["macro_f1"],
    }


def _top_k_labels(score_matrix: np.ndarray, classes: np.ndarray, top_k: int) -> list[list[str]]:
    top_indices = np.argsort(score_matrix, axis=1)[:, -top_k:][:, ::-1]
    return [[str(classes[index]) for index in row] for row in top_indices]


def _true_rank(score_matrix: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    ranks = []
    for row_index, class_index in enumerate(y_true):
        ordering = np.argsort(score_matrix[row_index])[::-1]
        ranks.append(int(np.where(ordering == class_index)[0][0]) + 1)
    return np.asarray(ranks)


def _top_k_accuracy(score_matrix: np.ndarray, y_true: np.ndarray, k: int) -> float:
    effective_k = min(k, score_matrix.shape[1])
    top_indices = np.argsort(score_matrix, axis=1)[:, -effective_k:]
    hits = [int(true_label in row) for true_label, row in zip(y_true, top_indices, strict=False)]
    return float(np.mean(hits))


def build_confusion_pairs(prediction_frame: pd.DataFrame, *, top_n: int = 12) -> pd.DataFrame:
    mistakes = prediction_frame.loc[prediction_frame["true_journal"] != prediction_frame["predicted_journal"]]
    if mistakes.empty:
        return pd.DataFrame(columns=["true_journal", "predicted_journal", "count"])
    confusion = (
        mistakes.groupby(["true_journal", "predicted_journal"])
        .size()
        .reset_index(name="count")
        .sort_values(["count", "true_journal"], ascending=[False, True])
        .head(top_n)
        .reset_index(drop=True)
    )
    return confusion


class JournalRecommender:
    def __init__(
        self,
        *,
        text_column: str = "combined_text",
        classifier_weight: float = 0.7,
        similarity_weight: float = 0.3,
        n_neighbors: int = 20,
        candidate_top_n: int | None = 5,
        classifier_c: float = 1.5,
        sublinear_tf: bool = True,
        min_df: int = 3,
        max_df: float = 0.9,
        max_features: int = 50000,
        random_state: int = DEFAULT_RANDOM_STATE,
    ) -> None:
        self.text_column = text_column
        self.classifier_weight = classifier_weight
        self.similarity_weight = similarity_weight
        self.n_neighbors = n_neighbors
        self.candidate_top_n = candidate_top_n
        self.classifier_c = classifier_c
        self.sublinear_tf = sublinear_tf
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, frame: pd.DataFrame) -> "JournalRecommender":
        required = {"journal", self.text_column}
        missing = required.difference(frame.columns)
        if missing:
            raise KeyError(f"Missing columns for training: {sorted(missing)}")

        self.training_frame_ = frame.reset_index(drop=True).copy()
        self.encoder_ = LabelEncoder()
        self.y_train_ = self.encoder_.fit_transform(self.training_frame_["journal"])
        self.classes_ = self.encoder_.classes_
        self.vectorizer_ = TfidfVectorizer(
            stop_words="english",
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=(1, 2),
            max_features=self.max_features,
            sublinear_tf=self.sublinear_tf,
        )
        self.train_matrix_ = self.vectorizer_.fit_transform(self.training_frame_[self.text_column])
        self.classifier_ = LinearSVC(C=self.classifier_c, random_state=self.random_state)
        self.classifier_.fit(self.train_matrix_, self.y_train_)
        self.neighbor_model_ = NearestNeighbors(
            n_neighbors=min(self.n_neighbors, len(self.training_frame_)),
            metric="cosine",
            algorithm="brute",
        )
        self.neighbor_model_.fit(self.train_matrix_)
        return self

    def _normalize_classifier_scores(self, raw_scores: np.ndarray) -> np.ndarray:
        raw_scores = np.atleast_2d(raw_scores)
        return _normalize_score_rows(raw_scores)

    def _prune_classifier_scores(self, normalized_scores: np.ndarray, raw_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        if self.candidate_top_n is None:
            return normalized_scores, None
        top_n = min(self.candidate_top_n, normalized_scores.shape[1])
        candidate_indices = np.argpartition(raw_scores, -top_n, axis=1)[:, -top_n:]
        pruned_scores = np.zeros_like(normalized_scores)
        row_indices = np.arange(normalized_scores.shape[0])[:, None]
        pruned_scores[row_indices, candidate_indices] = normalized_scores[row_indices, candidate_indices]
        pruned_scores = pruned_scores / np.maximum(pruned_scores.sum(axis=1, keepdims=True), 1e-9)
        return pruned_scores, candidate_indices

    def _similarity_vote(
        self,
        query_matrix: Any,
        *,
        candidate_indices: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[list[tuple[int, float]]]]:
        distances, neighbor_indices = self.neighbor_model_.kneighbors(query_matrix, return_distance=True)
        similarities = 1.0 - distances
        score_matrix = np.zeros((query_matrix.shape[0], len(self.classes_)), dtype=np.float32)
        neighbor_info: list[list[tuple[int, float]]] = []

        for row_index in range(query_matrix.shape[0]):
            candidate_set = None
            if candidate_indices is not None:
                candidate_set = set(int(item) for item in candidate_indices[row_index].tolist())

            weights = np.maximum(similarities[row_index], 0.0)
            neighbor_row_info: list[tuple[int, float]] = []
            total = 0.0
            for weight, neighbor_index in zip(weights, neighbor_indices[row_index]):
                class_index = int(self.y_train_[neighbor_index])
                if candidate_set is not None and class_index not in candidate_set:
                    continue
                score_matrix[row_index, class_index] += float(weight)
                total += float(weight)
                neighbor_row_info.append((int(neighbor_index), float(weight)))

            if total > 0:
                score_matrix[row_index] /= total
            neighbor_info.append(neighbor_row_info)

        return score_matrix, neighbor_info

    def _score_query_texts(self, query_texts: list[str]) -> tuple[np.ndarray, np.ndarray, list[list[tuple[int, float]]]]:
        query_matrix = self.vectorizer_.transform(query_texts)
        raw_scores = self.classifier_.decision_function(query_matrix)
        raw_scores = np.asarray(raw_scores)
        if raw_scores.ndim == 1:
            raw_scores = np.column_stack([-raw_scores, raw_scores])
        raw_scores = np.atleast_2d(raw_scores)
        classifier_scores = self._normalize_classifier_scores(raw_scores)
        classifier_scores, candidate_indices = self._prune_classifier_scores(classifier_scores, raw_scores)
        similarity_scores, neighbor_info = self._similarity_vote(query_matrix, candidate_indices=candidate_indices)

        if self.similarity_weight == 0:
            fused_scores = classifier_scores
        elif self.classifier_weight == 0:
            fused_scores = similarity_scores
        else:
            fused_scores = (self.classifier_weight * classifier_scores) + (self.similarity_weight * similarity_scores)
        return fused_scores, raw_scores, neighbor_info

    def predict_rankings(self, query_texts: Iterable[str], *, top_k: int = 5) -> tuple[list[list[str]], np.ndarray]:
        cleaned_queries = [clean_text(text) for text in query_texts]
        fused_scores, _, _ = self._score_query_texts(cleaned_queries)
        rankings = _top_k_labels(fused_scores, self.classes_, top_k)
        return rankings, fused_scores

    def score_queries(self, query_texts: Iterable[str]) -> np.ndarray:
        cleaned_queries = [clean_text(text) for text in query_texts]
        fused_scores, _, _ = self._score_query_texts(cleaned_queries)
        return fused_scores

    def _build_recommendation_payload(
        self,
        class_index: int,
        score: float,
        supporting_neighbors: list[pd.Series],
    ) -> dict[str, Any]:
        evidence_titles = [str(row["title"]) for row in supporting_neighbors]
        evidence_snippets = [_shorten_snippet(row.get("abstract", "")) for row in supporting_neighbors]
        subject_counter = Counter()
        keyword_counter = Counter()
        cluster_counter = Counter()
        for row in supporting_neighbors:
            subject_counter.update(row["subject_list"])
            keyword_counter.update(row["keyword_list"])
            cluster_label = str(row.get("cluster_label", "")).strip()
            if cluster_label:
                cluster_counter.update([cluster_label])

        evidence_subjects = [name for name, _ in subject_counter.most_common(3)]
        top_keywords = [name for name, _ in keyword_counter.most_common(5)]
        cluster_label = cluster_counter.most_common(1)[0][0] if cluster_counter else "Not available"
        keyword_text = ", ".join(top_keywords[:3]) if top_keywords else "related journal terms"
        explanation = (
            f"Recommended because similar articles contain keywords such as {keyword_text}. "
            f"Cluster: {cluster_label}."
        )

        return {
            "journal": str(self.classes_[class_index]),
            "score": round(float(score), 4),
            "confidence_score": round(float(score), 4),
            "top_keywords": top_keywords,
            "evidence_titles": evidence_titles,
            "evidence_snippets": evidence_snippets,
            "evidence_subjects": evidence_subjects,
            "cluster_label": cluster_label,
            "explanation": explanation,
        }

    def recommend(self, abstract_text: str, *, top_k: int = 5) -> list[dict[str, Any]]:
        query_text = clean_text(abstract_text)
        fused_scores, _, neighbor_info = self._score_query_texts([query_text])
        top_indices = np.argsort(fused_scores[0])[-top_k:][::-1]
        recommendations: list[dict[str, Any]] = []

        for class_index in top_indices:
            supporting_neighbors = [
                self.training_frame_.iloc[item_index]
                for item_index, _ in neighbor_info[0]
                if self.y_train_[item_index] == class_index
            ][:3]
            recommendations.append(self._build_recommendation_payload(class_index, fused_scores[0, class_index], supporting_neighbors))

        return recommendations

    def evaluate(
        self,
        frame: pd.DataFrame,
        *,
        query_column: str = "abstract_query_text",
        name: str = "model",
    ) -> EvaluationResult:
        if query_column not in frame.columns:
            raise KeyError(f"Missing query column: {query_column}")

        query_texts = frame[query_column].fillna("").map(clean_text).tolist()
        fused_scores, _, _ = self._score_query_texts(query_texts)
        true_labels = self.encoder_.transform(frame["journal"])
        predicted_labels = fused_scores.argmax(axis=1)
        top_5_predictions = _top_k_labels(fused_scores, self.classes_, 5)
        true_ranks = _true_rank(fused_scores, true_labels)
        sorted_indices = np.argsort(fused_scores, axis=1)[:, ::-1]
        predicted_scores = fused_scores[np.arange(len(predicted_labels)), predicted_labels]
        runner_up_indices = sorted_indices[:, 1] if fused_scores.shape[1] > 1 else sorted_indices[:, 0]
        runner_up_scores = (
            fused_scores[np.arange(len(predicted_labels)), runner_up_indices]
            if fused_scores.shape[1] > 1
            else np.zeros(len(predicted_labels), dtype=np.float32)
        )
        true_scores = fused_scores[np.arange(len(true_labels)), true_labels]
        metrics = {
            "top_1_accuracy": round(float(accuracy_score(true_labels, predicted_labels)), 4),
            "top_3_accuracy": round(_top_k_accuracy(fused_scores, true_labels, 3), 4),
            "top_5_accuracy": round(_top_k_accuracy(fused_scores, true_labels, 5), 4),
            "macro_f1": round(float(f1_score(true_labels, predicted_labels, average="macro")), 4),
        }

        predictions = pd.DataFrame(
            {
                "record_id": frame["record_id"].to_numpy(),
                "title": frame["title"].to_numpy(),
                "true_journal": frame["journal"].to_numpy(),
                "predicted_journal": self.classes_[predicted_labels],
                "runner_up_journal": self.classes_[runner_up_indices],
                "predicted_confidence": np.round(predicted_scores, 4),
                "runner_up_confidence": np.round(runner_up_scores, 4),
                "score_margin": np.round(predicted_scores - runner_up_scores, 4),
                "true_journal_score": np.round(true_scores, 4),
                "true_rank": true_ranks,
                "is_top_1": true_ranks == 1,
                "is_top_5": true_ranks <= 5,
                "top_5_journals": [" | ".join(row) for row in top_5_predictions],
            }
        )
        confusion_pairs = build_confusion_pairs(predictions)
        return EvaluationResult(name=name, metrics=metrics, predictions=predictions, confusion_pairs=confusion_pairs)


class SemanticJournalRecommender:
    def __init__(
        self,
        *,
        text_column: str = "combined_text",
        model_name: str = DEFAULT_SBERT_MODEL_NAME,
        embedding_cache_dir: str | Path = DEFAULT_CACHE_DIR,
        model_cache_dir: str | Path | None = None,
        n_neighbors: int = 30,
        batch_size: int = 64,
        scoring_batch_size: int = 128,
        random_state: int = DEFAULT_RANDOM_STATE,
    ) -> None:
        self.text_column = text_column
        self.model_name = model_name
        self.embedding_cache_dir = Path(embedding_cache_dir)
        self.model_cache_dir = Path(model_cache_dir) if model_cache_dir is not None else self.embedding_cache_dir / "models"
        self.n_neighbors = n_neighbors
        self.batch_size = batch_size
        self.scoring_batch_size = scoring_batch_size
        self.random_state = random_state

    def _load_model(self) -> SentenceTransformer:
        if not hasattr(self, "model_"):
            self.model_cache_dir.mkdir(parents=True, exist_ok=True)
            local_model_path = None
            model_roots = [self.model_cache_dir / f"models--{self.model_name.replace('/', '--')}"]
            model_roots.extend(
                sorted(
                    [
                        path
                        for path in self.model_cache_dir.glob(f"models--*{self.model_name.replace('/', '--')}")
                        if path.is_dir()
                    ]
                )
            )
            for model_root in model_roots:
                snapshots_dir = model_root / "snapshots"
                if not snapshots_dir.exists():
                    continue
                snapshots = sorted([path for path in snapshots_dir.iterdir() if path.is_dir()])
                if snapshots:
                    local_model_path = str(snapshots[-1])
                    break

            model_source = local_model_path or self.model_name
            self.model_ = SentenceTransformer(
                model_source,
                cache_folder=str(self.model_cache_dir),
                local_files_only=local_model_path is not None,
                model_kwargs={"local_files_only": local_model_path is not None},
                processor_kwargs={"local_files_only": local_model_path is not None},
                config_kwargs={"local_files_only": local_model_path is not None},
            )
        return self.model_

    def _cache_key(self, frame: pd.DataFrame) -> str:
        hasher = hashlib.sha256()
        hasher.update(self.model_name.encode("utf-8"))
        hasher.update(self.text_column.encode("utf-8"))
        for row in frame[["record_id", self.text_column]].itertuples(index=False):
            hasher.update(str(row[0]).encode("utf-8"))
            hasher.update(str(row[1]).encode("utf-8"))
        return hasher.hexdigest()[:20]

    def _embedding_cache_path(self, frame: pd.DataFrame) -> Path:
        cache_key = self._cache_key(frame)
        self.embedding_cache_dir.mkdir(parents=True, exist_ok=True)
        safe_model_name = self.model_name.replace("/", "_")
        return self.embedding_cache_dir / f"sbert_{safe_model_name}_{cache_key}.npz"

    def _encode_texts(self, texts: list[str], *, cache_path: Path | None = None) -> np.ndarray:
        if cache_path is not None and cache_path.exists():
            with np.load(cache_path) as cached:
                return cached["embeddings"]

        model = self._load_model()
        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if cache_path is not None:
            np.savez_compressed(cache_path, embeddings=embeddings)
        return embeddings

    def fit(self, frame: pd.DataFrame) -> "SemanticJournalRecommender":
        required = {"journal", self.text_column}
        missing = required.difference(frame.columns)
        if missing:
            raise KeyError(f"Missing columns for training: {sorted(missing)}")

        self.training_frame_ = frame.reset_index(drop=True).copy()
        self.encoder_ = LabelEncoder()
        self.y_train_ = self.encoder_.fit_transform(self.training_frame_["journal"])
        self.classes_ = self.encoder_.classes_
        cache_path = self._embedding_cache_path(self.training_frame_)
        texts = self.training_frame_[self.text_column].fillna("").map(clean_text).tolist()
        self.train_embeddings_ = self._encode_texts(texts, cache_path=cache_path)
        return self

    def _score_query_texts(self, query_texts: list[str]) -> tuple[np.ndarray, list[list[tuple[int, float]]]]:
        cleaned_queries = [clean_text(text) for text in query_texts]
        query_embeddings = self._encode_texts(cleaned_queries, cache_path=None)
        score_matrix = np.zeros((len(query_texts), len(self.classes_)), dtype=np.float32)
        neighbor_info: list[list[tuple[int, float]]] = []

        top_n = min(self.n_neighbors, len(self.training_frame_))
        for batch_start in range(0, len(query_texts), self.scoring_batch_size):
            batch_end = min(batch_start + self.scoring_batch_size, len(query_texts))
            similarity_batch = query_embeddings[batch_start:batch_end] @ self.train_embeddings_.T

            for local_index, similarity_row in enumerate(similarity_batch):
                row_index = batch_start + local_index
                candidate_indices = np.argsort(similarity_row)[-top_n:][::-1]
                row_neighbors = [(int(item_index), float(similarity_row[item_index])) for item_index in candidate_indices]
                neighbor_info.append(row_neighbors)

                journal_frequency = Counter()
                journal_similarity_sum: dict[int, float] = {}
                for item_index in candidate_indices:
                    class_index = int(self.y_train_[item_index])
                    journal_frequency[class_index] += 1
                    journal_similarity_sum[class_index] = journal_similarity_sum.get(class_index, 0.0) + float(
                        similarity_row[item_index]
                    )

                if not journal_frequency:
                    continue

                max_frequency = max(journal_frequency.values())
                mean_similarity_values = {
                    class_index: journal_similarity_sum[class_index] / journal_frequency[class_index]
                    for class_index in journal_frequency
                }
                max_similarity = max(mean_similarity_values.values())
                min_similarity = min(mean_similarity_values.values())

                for class_index, frequency in journal_frequency.items():
                    frequency_score = frequency / max_frequency
                    mean_similarity = mean_similarity_values[class_index]
                    similarity_score = (mean_similarity - min_similarity) / max(max_similarity - min_similarity, 1e-9)
                    score_matrix[row_index, class_index] = 0.5 * frequency_score + 0.5 * similarity_score

        score_matrix = _normalize_score_rows(score_matrix)
        return score_matrix, neighbor_info

    def score_queries(self, query_texts: Iterable[str]) -> np.ndarray:
        score_matrix, _ = self._score_query_texts(list(query_texts))
        return score_matrix

    def _build_recommendation_payload(
        self,
        class_index: int,
        score: float,
        neighbors: list[tuple[int, float]],
    ) -> dict[str, Any]:
        supporting_rows = [
            self.training_frame_.iloc[item_index]
            for item_index, _ in neighbors
            if int(self.y_train_[item_index]) == class_index
        ][:3]
        evidence_titles = [str(row["title"]) for row in supporting_rows]
        evidence_snippets = [_shorten_snippet(row.get("abstract", "")) for row in supporting_rows]
        keyword_counter = Counter()
        subject_counter = Counter()
        cluster_counter = Counter()
        for row in supporting_rows:
            keyword_counter.update(row["keyword_list"])
            subject_counter.update(row["subject_list"])
            cluster_label = str(row.get("cluster_label", "")).strip()
            if cluster_label:
                cluster_counter.update([cluster_label])

        top_keywords = [name for name, _ in keyword_counter.most_common(5)]
        evidence_subjects = [name for name, _ in subject_counter.most_common(3)]
        cluster_label = cluster_counter.most_common(1)[0][0] if cluster_counter else "Not available"
        keyword_text = ", ".join(top_keywords[:3]) if top_keywords else "related journal terms"
        explanation = (
            f"Recommended because similar articles contain keywords such as {keyword_text}. "
            f"Cluster: {cluster_label}."
        )

        return {
            "journal": str(self.classes_[class_index]),
            "score": round(float(score), 4),
            "confidence_score": round(float(score), 4),
            "top_keywords": top_keywords,
            "evidence_titles": evidence_titles,
            "evidence_snippets": evidence_snippets,
            "evidence_subjects": evidence_subjects,
            "cluster_label": cluster_label,
            "explanation": explanation,
        }

    def recommend(self, input_abstract: str, *, top_k: int = 5) -> list[dict[str, Any]]:
        score_matrix, neighbor_info = self._score_query_texts([input_abstract])
        top_indices = np.argsort(score_matrix[0])[-top_k:][::-1]
        return [
            self._build_recommendation_payload(class_index, score_matrix[0, class_index], neighbor_info[0])
            for class_index in top_indices
        ]

    def recommend_journals_bert(self, input_abstract: str, top_n: int = 5) -> list[dict[str, Any]]:
        return self.recommend(input_abstract, top_k=top_n)

    def evaluate(
        self,
        frame: pd.DataFrame,
        *,
        query_column: str = "abstract_query_text",
        name: str = "BERT model",
    ) -> EvaluationResult:
        if query_column not in frame.columns:
            raise KeyError(f"Missing query column: {query_column}")

        query_texts = frame[query_column].fillna("").tolist()
        score_matrix, _ = self._score_query_texts(query_texts)
        true_labels = self.encoder_.transform(frame["journal"])
        predicted_labels = score_matrix.argmax(axis=1)
        top_predictions = _top_k_labels(score_matrix, self.classes_, 5)
        true_ranks = _true_rank(score_matrix, true_labels)
        sorted_indices = np.argsort(score_matrix, axis=1)[:, ::-1]
        predicted_scores = score_matrix[np.arange(len(predicted_labels)), predicted_labels]
        runner_up_indices = sorted_indices[:, 1] if score_matrix.shape[1] > 1 else sorted_indices[:, 0]
        runner_up_scores = (
            score_matrix[np.arange(len(predicted_labels)), runner_up_indices]
            if score_matrix.shape[1] > 1
            else np.zeros(len(predicted_labels), dtype=np.float32)
        )
        true_scores = score_matrix[np.arange(len(true_labels)), true_labels]

        metrics = {
            "top_1_accuracy": round(float(accuracy_score(true_labels, predicted_labels)), 4),
            "top_3_accuracy": round(_top_k_accuracy(score_matrix, true_labels, 3), 4),
            "top_5_accuracy": round(_top_k_accuracy(score_matrix, true_labels, 5), 4),
            "macro_f1": round(float(f1_score(true_labels, predicted_labels, average="macro")), 4),
        }

        predictions = pd.DataFrame(
            {
                "record_id": frame["record_id"].to_numpy(),
                "title": frame["title"].to_numpy(),
                "true_journal": frame["journal"].to_numpy(),
                "predicted_journal": self.classes_[predicted_labels],
                "runner_up_journal": self.classes_[runner_up_indices],
                "predicted_confidence": np.round(predicted_scores, 4),
                "runner_up_confidence": np.round(runner_up_scores, 4),
                "score_margin": np.round(predicted_scores - runner_up_scores, 4),
                "true_journal_score": np.round(true_scores, 4),
                "true_rank": true_ranks,
                "is_top_1": true_ranks == 1,
                "is_top_5": true_ranks <= 5,
                "top_5_journals": [" | ".join(row) for row in top_predictions],
            }
        )
        confusion_pairs = build_confusion_pairs(predictions)
        return EvaluationResult(name=name, metrics=metrics, predictions=predictions, confusion_pairs=confusion_pairs)


class EnsembleJournalRecommender:
    def __init__(
        self,
        tfidf_model: JournalRecommender,
        bert_model: SemanticJournalRecommender,
        *,
        tfidf_weight: float = 0.5,
        bert_weight: float = 0.5,
    ) -> None:
        self.tfidf_model = tfidf_model
        self.bert_model = bert_model
        self.tfidf_weight = tfidf_weight
        self.bert_weight = bert_weight

    def fit(self, frame: pd.DataFrame) -> "EnsembleJournalRecommender":
        self.tfidf_model.fit(frame)
        self.bert_model.fit(frame)
        if list(self.tfidf_model.classes_) != list(self.bert_model.classes_):
            raise ValueError("TF-IDF and BERT models must share the same journal label order for ensemble scoring.")
        self.classes_ = self.tfidf_model.classes_
        self.training_frame_ = frame.reset_index(drop=True).copy()
        return self

    def _score_query_texts(self, query_texts: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tfidf_scores = self.tfidf_model.score_queries(query_texts)
        bert_scores = self.bert_model.score_queries(query_texts)
        ensemble_scores = (self.tfidf_weight * tfidf_scores) + (self.bert_weight * bert_scores)
        ensemble_scores = ensemble_scores / np.maximum(ensemble_scores.sum(axis=1, keepdims=True), 1e-9)
        return ensemble_scores, tfidf_scores, bert_scores

    def score_queries(self, query_texts: Iterable[str]) -> np.ndarray:
        ensemble_scores, _, _ = self._score_query_texts(list(query_texts))
        return ensemble_scores

    def recommend(self, input_abstract: str, *, top_k: int = 5) -> list[dict[str, Any]]:
        ensemble_scores, tfidf_scores, bert_scores = self._score_query_texts([input_abstract])
        top_indices = np.argsort(ensemble_scores[0])[-top_k:][::-1]
        bert_recommendations = {item["journal"]: item for item in self.bert_model.recommend(input_abstract, top_k=top_k)}
        tfidf_recommendations = {item["journal"]: item for item in self.tfidf_model.recommend(input_abstract, top_k=top_k)}
        payload: list[dict[str, Any]] = []
        for class_index in top_indices:
            journal = str(self.classes_[class_index])
            bert_item = bert_recommendations.get(journal, {})
            tfidf_item = tfidf_recommendations.get(journal, {})
            top_keywords = bert_item.get("top_keywords") or tfidf_item.get("top_keywords") or []
            evidence_titles = bert_item.get("evidence_titles") or tfidf_item.get("evidence_titles") or []
            evidence_snippets = bert_item.get("evidence_snippets") or tfidf_item.get("evidence_snippets") or []
            cluster_label = bert_item.get("cluster_label") or tfidf_item.get("cluster_label") or "Not available"
            keyword_text = ", ".join(top_keywords[:3]) if top_keywords else "related journal terms"
            payload.append(
                {
                    "journal": journal,
                    "score": round(float(ensemble_scores[0, class_index]), 4),
                    "confidence_score": round(float(ensemble_scores[0, class_index]), 4),
                    "tfidf_score": round(float(tfidf_scores[0, class_index]), 4),
                    "bert_score": round(float(bert_scores[0, class_index]), 4),
                    "top_keywords": top_keywords,
                    "evidence_titles": evidence_titles,
                    "evidence_snippets": evidence_snippets,
                    "evidence_subjects": bert_item.get("evidence_subjects") or tfidf_item.get("evidence_subjects") or [],
                    "cluster_label": cluster_label,
                    "explanation": (
                        f"Recommended because similar articles contain keywords such as {keyword_text}. "
                        f"Cluster: {cluster_label}."
                    ),
                }
            )
        return payload

    def recommend_journals_ensemble(self, input_abstract: str, top_n: int = 5) -> list[dict[str, Any]]:
        return self.recommend(input_abstract, top_k=top_n)

    def evaluate(
        self,
        frame: pd.DataFrame,
        *,
        query_column: str = "abstract_query_text",
        name: str = "Ensemble model",
    ) -> EvaluationResult:
        if query_column not in frame.columns:
            raise KeyError(f"Missing query column: {query_column}")

        query_texts = frame[query_column].fillna("").tolist()
        ensemble_scores, _, _ = self._score_query_texts(query_texts)
        true_labels = self.tfidf_model.encoder_.transform(frame["journal"])
        predicted_labels = ensemble_scores.argmax(axis=1)
        top_predictions = _top_k_labels(ensemble_scores, self.classes_, 5)
        true_ranks = _true_rank(ensemble_scores, true_labels)
        sorted_indices = np.argsort(ensemble_scores, axis=1)[:, ::-1]
        predicted_scores = ensemble_scores[np.arange(len(predicted_labels)), predicted_labels]
        runner_up_indices = sorted_indices[:, 1] if ensemble_scores.shape[1] > 1 else sorted_indices[:, 0]
        runner_up_scores = (
            ensemble_scores[np.arange(len(predicted_labels)), runner_up_indices]
            if ensemble_scores.shape[1] > 1
            else np.zeros(len(predicted_labels), dtype=np.float32)
        )
        true_scores = ensemble_scores[np.arange(len(true_labels)), true_labels]

        metrics = {
            "top_1_accuracy": round(float(accuracy_score(true_labels, predicted_labels)), 4),
            "top_3_accuracy": round(_top_k_accuracy(ensemble_scores, true_labels, 3), 4),
            "top_5_accuracy": round(_top_k_accuracy(ensemble_scores, true_labels, 5), 4),
            "macro_f1": round(float(f1_score(true_labels, predicted_labels, average="macro")), 4),
        }

        predictions = pd.DataFrame(
            {
                "record_id": frame["record_id"].to_numpy(),
                "title": frame["title"].to_numpy(),
                "true_journal": frame["journal"].to_numpy(),
                "predicted_journal": self.classes_[predicted_labels],
                "runner_up_journal": self.classes_[runner_up_indices],
                "predicted_confidence": np.round(predicted_scores, 4),
                "runner_up_confidence": np.round(runner_up_scores, 4),
                "score_margin": np.round(predicted_scores - runner_up_scores, 4),
                "true_journal_score": np.round(true_scores, 4),
                "true_rank": true_ranks,
                "is_top_1": true_ranks == 1,
                "is_top_5": true_ranks <= 5,
                "top_5_journals": [" | ".join(row) for row in top_predictions],
            }
        )
        confusion_pairs = build_confusion_pairs(predictions)
        return EvaluationResult(name=name, metrics=metrics, predictions=predictions, confusion_pairs=confusion_pairs)


def recommend_journals_bert(
    model: SemanticJournalRecommender,
    input_abstract: str,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    return model.recommend_journals_bert(input_abstract, top_n=top_n)


def recommend_journals_ensemble(
    model: EnsembleJournalRecommender,
    input_abstract: str,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    return model.recommend_journals_ensemble(input_abstract, top_n=top_n)


class TopicClusterer:
    def __init__(
        self,
        *,
        text_column: str = "combined_text",
        candidate_clusters: tuple[int, ...] = tuple(range(10, 61)),
        svd_components: int = 100,
        max_features: int = 40000,
        min_df: int = 5,
        max_df: float = 0.85,
        silhouette_tolerance: float = 0.02,
        silhouette_sample_size: int = 3000,
        selection_strategy: str = "max_silhouette",
        random_state: int = DEFAULT_RANDOM_STATE,
    ) -> None:
        self.text_column = text_column
        self.candidate_clusters = candidate_clusters
        self.svd_components = svd_components
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.silhouette_tolerance = silhouette_tolerance
        self.silhouette_sample_size = silhouette_sample_size
        self.selection_strategy = selection_strategy
        self.random_state = random_state

    def fit(self, frame: pd.DataFrame) -> "TopicClusterer":
        self.frame_ = frame.reset_index(drop=True).copy()
        self.vectorizer_ = TfidfVectorizer(
            stop_words="english",
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=(1, 2),
            max_features=self.max_features,
            sublinear_tf=True,
        )
        self.term_matrix_ = self.vectorizer_.fit_transform(self.frame_[self.text_column])
        max_components = max(2, min(self.svd_components, self.term_matrix_.shape[0] - 1, self.term_matrix_.shape[1] - 1))
        self.svd_ = TruncatedSVD(n_components=max_components, random_state=self.random_state)
        reduced = self.svd_.fit_transform(self.term_matrix_)
        self.normalizer_ = Normalizer(copy=False)
        self.reduced_matrix_ = self.normalizer_.fit_transform(reduced)

        rng = np.random.default_rng(self.random_state)
        sample_size = min(self.silhouette_sample_size, len(self.frame_))
        sample_indices = rng.choice(len(self.frame_), size=sample_size, replace=False)

        silhouette_rows = []
        best_score = -1.0
        candidate_scores: dict[int, float] = {}
        fitted_models: dict[int, KMeans] = {}

        for cluster_count in self.candidate_clusters:
            model = KMeans(n_clusters=cluster_count, n_init=20, random_state=self.random_state)
            labels = model.fit_predict(self.reduced_matrix_)
            sample_score = silhouette_score(self.reduced_matrix_[sample_indices], labels[sample_indices])
            silhouette_rows.append({"k": cluster_count, "silhouette": round(float(sample_score), 4)})
            candidate_scores[cluster_count] = float(sample_score)
            fitted_models[cluster_count] = model
            best_score = max(best_score, float(sample_score))

        if self.selection_strategy == "max_silhouette":
            best_candidates = [
                cluster_count for cluster_count, score in candidate_scores.items() if np.isclose(score, best_score)
            ]
            self.best_k_ = min(best_candidates)
        elif self.selection_strategy == "smallest_within_tolerance":
            viable_clusters = [
                cluster_count
                for cluster_count, score in candidate_scores.items()
                if score >= best_score - self.silhouette_tolerance
            ]
            self.best_k_ = min(viable_clusters)
        else:
            raise ValueError(f"Unsupported cluster selection strategy: {self.selection_strategy}")

        self.best_silhouette_ = float(candidate_scores[self.best_k_])
        self.kmeans_ = fitted_models[self.best_k_]
        self.labels_ = self.kmeans_.labels_
        self.silhouette_table_ = pd.DataFrame(silhouette_rows).sort_values("k").reset_index(drop=True)
        self.silhouette_table_["selected"] = self.silhouette_table_["k"].eq(self.best_k_)

        self.projection_2d_ = PCA(n_components=2, random_state=self.random_state).fit_transform(self.reduced_matrix_)
        reconstructed_centers = self.kmeans_.cluster_centers_ @ self.svd_.components_
        feature_names = np.asarray(self.vectorizer_.get_feature_names_out())

        cluster_rows: list[dict[str, Any]] = []
        for cluster_id in range(self.best_k_):
            cluster_mask = self.labels_ == cluster_id
            cluster_frame = self.frame_.loc[cluster_mask].copy()
            top_term_indices = np.argsort(reconstructed_centers[cluster_id])[-10:][::-1]
            top_terms = feature_names[top_term_indices].tolist()
            subject_counter = Counter()
            for subject_list in cluster_frame["subject_list"]:
                subject_counter.update(subject_list)
            example_titles = cluster_frame["title"].head(3).tolist()
            cluster_rows.append(
                {
                    "cluster": cluster_id,
                    "size": int(cluster_mask.sum()),
                    "top_terms": ", ".join(top_terms[:10]),
                    "dominant_subjects": ", ".join(name for name, _ in subject_counter.most_common(5)),
                    "example_titles": " | ".join(example_titles),
                }
            )
        self.cluster_summary_ = pd.DataFrame(cluster_rows).sort_values("cluster").reset_index(drop=True)
        return self

    def summarize_clusters(self) -> pd.DataFrame:
        return self.cluster_summary_.copy()


def plot_publication_trend(frame: pd.DataFrame, *, ax: Axes | None = None) -> Axes:
    axis = ax or plt.subplots(figsize=(10, 4))[1]
    yearly_counts = frame.groupby("pub_year").size().sort_index()
    axis.plot(yearly_counts.index, yearly_counts.values, color="#0b7285", linewidth=2.5, marker="o", markersize=4)
    axis.set_title("Publication Trend by Year")
    axis.set_xlabel("Publication Year")
    axis.set_ylabel("Article Count")
    axis.grid(alpha=0.25)
    return axis


def plot_top_journals(frame: pd.DataFrame, *, top_n: int = 15, ax: Axes | None = None) -> Axes:
    axis = ax or plt.subplots(figsize=(10, 6))[1]
    journal_counts = frame["journal"].value_counts().head(top_n).sort_values()
    axis.barh(journal_counts.index, journal_counts.values, color="#5c7cfa")
    axis.set_title(f"Top {top_n} Journals by Article Count")
    axis.set_xlabel("Article Count")
    return axis


def plot_metric_comparison(metric_frame: pd.DataFrame, *, ax: Axes | None = None) -> Axes:
    axis = ax or plt.subplots(figsize=(8, 4))[1]
    metric_columns = ["top_1_accuracy", "top_3_accuracy", "top_5_accuracy", "macro_f1"]
    x_positions = np.arange(len(metric_columns))
    width = 0.22

    for offset, (_, row) in enumerate(metric_frame.iterrows()):
        values = [row[column] for column in metric_columns]
        axis.bar(x_positions + (offset - 1) * width, values, width=width, label=row["model"])

    axis.set_xticks(x_positions)
    axis.set_xticklabels(["Top-1", "Top-3", "Top-5", "Macro-F1"])
    axis.set_ylim(0, 1)
    axis.set_title("Model Comparison")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    return axis


def plot_cluster_projection(clusterer: TopicClusterer, *, ax: Axes | None = None) -> Axes:
    axis = ax or plt.subplots(figsize=(10, 6))[1]
    scatter = axis.scatter(
        clusterer.projection_2d_[:, 0],
        clusterer.projection_2d_[:, 1],
        c=clusterer.labels_,
        cmap="nipy_spectral",
        s=12,
        alpha=0.75,
    )
    axis.set_title(f"Topic Clusters Projection (k={clusterer.best_k_})")
    axis.set_xlabel("Projection 1")
    axis.set_ylabel("Projection 2")
    if clusterer.best_k_ <= 20:
        legend = axis.legend(*scatter.legend_elements(num=clusterer.best_k_), title="Cluster", loc="best")
        axis.add_artist(legend)
    else:
        axis.text(
            0.99,
            0.02,
            f"{clusterer.best_k_} fine-grained clusters",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "#adb5bd"},
        )
    return axis


def _sample_topic_text(clusterer: TopicClusterer, *, top_n: int = 3, top_terms: int = 4) -> list[str]:
    summary = clusterer.summarize_clusters().sort_values(["size", "cluster"], ascending=[False, True]).head(top_n)
    topic_lines: list[str] = []
    for _, row in summary.iterrows():
        terms = [term.strip() for term in str(row["top_terms"]).split(",") if term.strip()][:top_terms]
        topic_lines.append(f"Cluster {int(row['cluster'])}: {', '.join(terms)}")
    return topic_lines


def summarize_cluster_configurations(clusterers: Iterable[TopicClusterer]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for clusterer in clusterers:
        cluster_summary = clusterer.summarize_clusters()
        cluster_sizes = cluster_summary["size"].astype(int)
        rows.append(
            {
                "k": int(clusterer.best_k_),
                "silhouette": round(float(clusterer.best_silhouette_), 4),
                "mean_cluster_size": round(float(cluster_sizes.mean()), 2),
                "median_cluster_size": round(float(cluster_sizes.median()), 2),
                "min_cluster_size": int(cluster_sizes.min()),
                "max_cluster_size": int(cluster_sizes.max()),
                "example_topics": " | ".join(_sample_topic_text(clusterer)),
            }
        )
    return pd.DataFrame(rows).sort_values("k").reset_index(drop=True)


def plot_k_comparison(
    clusterer_a: TopicClusterer,
    clusterer_b: TopicClusterer,
    *,
    axes: np.ndarray | None = None,
) -> np.ndarray:
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(18, 5), gridspec_kw={"width_ratios": [1.0, 1.3, 1.7]})

    clusterers = [clusterer_a, clusterer_b]
    colors = ["#1c7ed6", "#e8590c"]
    labels = [f"k={clusterer.best_k_}" for clusterer in clusterers]
    summary = summarize_cluster_configurations(clusterers)

    axes[0].bar(labels, summary["silhouette"], color=colors, width=0.55)
    axes[0].set_ylim(0, max(summary["silhouette"]) * 1.15)
    axes[0].set_title("Silhouette Comparison")
    axes[0].set_ylabel("Silhouette Score")
    axes[0].grid(axis="y", alpha=0.2)

    for label, clusterer, color in zip(labels, clusterers, colors, strict=False):
        cluster_sizes = clusterer.summarize_clusters()["size"].sort_values(ascending=False).reset_index(drop=True)
        axes[1].plot(
            np.arange(1, len(cluster_sizes) + 1),
            cluster_sizes.to_numpy(),
            linewidth=2.2,
            color=color,
            label=label,
        )
    axes[1].set_title("Sorted Cluster Sizes")
    axes[1].set_xlabel("Cluster Rank")
    axes[1].set_ylabel("Articles per Cluster")
    axes[1].legend()
    axes[1].grid(alpha=0.2)

    axes[2].axis("off")
    y_position = 0.95
    for label, clusterer, color in zip(labels, clusterers, colors, strict=False):
        axes[2].text(
            0.0,
            y_position,
            f"{label} | silhouette={clusterer.best_silhouette_:.4f}",
            fontsize=11,
            fontweight="bold",
            color=color,
            transform=axes[2].transAxes,
        )
        y_position -= 0.08
        for topic_text in _sample_topic_text(clusterer):
            axes[2].text(
                0.02,
                y_position,
                f"- {topic_text}",
                fontsize=10,
                transform=axes[2].transAxes,
                wrap=True,
            )
            y_position -= 0.08
        y_position -= 0.06
    axes[2].set_title("Representative Topic Examples")
    return axes


def select_demo_examples(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    n_examples: int = 3,
) -> pd.DataFrame:
    merged = frame.merge(predictions[["record_id", "is_top_5", "true_rank"]], on="record_id", how="left")
    candidates = merged.loc[merged["is_top_5"]].copy()
    candidates["primary_subject"] = candidates["subject_list"].map(lambda values: values[0] if values else "Unknown")
    chosen_rows = []
    seen_subjects: set[str] = set()

    for _, row in candidates.sort_values(["true_rank", "pub_year"]).iterrows():
        subject = row["primary_subject"]
        if subject in seen_subjects:
            continue
        seen_subjects.add(subject)
        chosen_rows.append(row)
        if len(chosen_rows) == n_examples:
            break

    if len(chosen_rows) < n_examples:
        remaining = candidates.loc[~candidates["record_id"].isin([row["record_id"] for row in chosen_rows])].head(
            n_examples - len(chosen_rows)
        )
        chosen_rows.extend(list(remaining.to_dict(orient="records")))

    return pd.DataFrame(chosen_rows).reset_index(drop=True)


def run_ablation_study(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    query_column: str = "abstract_query_text",
    classifier_c: float = 0.5,
    sublinear_tf: bool = True,
) -> pd.DataFrame:
    ablation_configs = [
        ("abstract only", "abstract_only_text"),
        ("title + abstract", "title_abstract_text"),
        ("abstract + keywords", "abstract_keywords_text"),
        ("abstract + keywords + subjects", "abstract_keywords_subjects_text"),
        ("full combined_text", "combined_text"),
    ]

    rows: list[dict[str, Any]] = []
    for label, text_column in ablation_configs:
        model = JournalRecommender(
            text_column=text_column,
            classifier_weight=1.0,
            similarity_weight=0.0,
            candidate_top_n=None,
            classifier_c=classifier_c,
            sublinear_tf=sublinear_tf,
            n_neighbors=20,
        )
        model.fit(train_frame)
        result = model.evaluate(test_frame, query_column=query_column, name=label)
        rows.append(
            {
                "representation": label,
                "text_column": text_column,
                "top_1_accuracy": result.metrics["top_1_accuracy"],
                "top_3_accuracy": result.metrics["top_3_accuracy"],
                "top_5_accuracy": result.metrics["top_5_accuracy"],
                "macro_f1": result.metrics["macro_f1"],
            }
        )

    return pd.DataFrame(rows).sort_values(["top_5_accuracy", "top_1_accuracy"], ascending=[False, False]).reset_index(
        drop=True
    )


def compute_per_journal_top5(
    predictions: pd.DataFrame,
    *,
    min_test_samples: int = 5,
) -> pd.DataFrame:
    per_journal = (
        predictions.groupby("true_journal")
        .agg(
            test_samples=("record_id", "size"),
            top_1_accuracy=("is_top_1", "mean"),
            top_5_hit_rate=("is_top_5", "mean"),
            median_true_rank=("true_rank", "median"),
        )
        .reset_index()
    )
    per_journal["top_1_accuracy"] = per_journal["top_1_accuracy"].round(4)
    per_journal["top_5_hit_rate"] = per_journal["top_5_hit_rate"].round(4)
    eligible = per_journal.loc[per_journal["test_samples"] >= min_test_samples].copy()
    eligible = eligible.sort_values(["top_5_hit_rate", "test_samples", "true_journal"], ascending=[False, False, True])
    return eligible.reset_index(drop=True)


def summarize_class_imbalance(frame: pd.DataFrame, *, document_type: str = "Article") -> tuple[pd.DataFrame, pd.DataFrame]:
    article_frame = frame.loc[frame["document_type"].fillna("").eq(document_type) & frame["abstract"].str.len().gt(0)].copy()
    counts = (
        article_frame.groupby("journal")
        .size()
        .reset_index(name="article_count")
        .sort_values(["article_count", "journal"], ascending=[False, True])
        .reset_index(drop=True)
    )
    summary = pd.DataFrame(
        [
            {
                "article_records_with_abstract": int(len(article_frame)),
                "journals_total": int(counts["journal"].nunique()),
                "journals_lt_5": int((counts["article_count"] < 5).sum()),
                "journals_lt_10": int((counts["article_count"] < 10).sum()),
                "journals_lt_20": int((counts["article_count"] < 20).sum()),
                "median_articles_per_journal": float(counts["article_count"].median()),
                "mean_articles_per_journal": round(float(counts["article_count"].mean()), 2),
                "max_articles_per_journal": int(counts["article_count"].max()),
            }
        ]
    )
    return counts, summary


def plot_ablation_results(ablation_frame: pd.DataFrame, *, ax: Axes | None = None) -> Axes:
    axis = ax or plt.subplots(figsize=(10, 5))[1]
    ordered = ablation_frame.copy()
    metrics = ["top_1_accuracy", "top_3_accuracy", "top_5_accuracy"]
    x_positions = np.arange(len(ordered))
    width = 0.22

    for offset, metric in enumerate(metrics):
        axis.bar(
            x_positions + (offset - 1) * width,
            ordered[metric].to_numpy(),
            width=width,
            label=metric.replace("_", " ").title(),
        )

    axis.set_xticks(x_positions)
    axis.set_xticklabels(ordered["representation"], rotation=20, ha="right")
    axis.set_ylim(0, 1)
    axis.set_title("Ablation Study Across Input Representations")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    return axis


def plot_journal_imbalance(counts_frame: pd.DataFrame, *, ax: Axes | None = None) -> Axes:
    axis = ax or plt.subplots(figsize=(10, 5))[1]
    sorted_counts = counts_frame["article_count"].sort_values(ascending=False).reset_index(drop=True)
    axis.plot(np.arange(1, len(sorted_counts) + 1), sorted_counts.to_numpy(), color="#e8590c", linewidth=2.0)
    axis.axhline(20, color="#2b8a3e", linestyle="--", linewidth=1, label="20 articles")
    axis.axhline(10, color="#1c7ed6", linestyle="--", linewidth=1, label="10 articles")
    axis.axhline(5, color="#c92a2a", linestyle="--", linewidth=1, label="5 articles")
    axis.set_title("Journal Class Imbalance (Ranked by Article Count)")
    axis.set_xlabel("Journal Rank")
    axis.set_ylabel("Article Count")
    axis.legend()
    axis.grid(alpha=0.2)
    return axis


def plot_system_pipeline(ax: Axes | None = None) -> Axes:
    axis = ax or plt.subplots(figsize=(15, 4))[1]
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    steps = [
        ("User\nAbstract", "#e3fafc"),
        ("Text\nPreprocessing", "#d3f9d8"),
        ("Feature Extraction\nTF-IDF / BERT", "#fff3bf"),
        ("Similarity\nComputation", "#ffd8a8"),
        ("Journal Ranking\nTF-IDF / BERT /\nHybrid / Ensemble", "#ffc9c9"),
        ("Top-5\nRecommendation", "#e5dbff"),
        ("Explainable\nOutput", "#d0ebff"),
    ]
    x_positions = np.linspace(0.08, 0.92, len(steps))
    box_width = 0.11
    box_height = 0.22
    y_position = 0.52

    for index, ((label, color), x_position) in enumerate(zip(steps, x_positions, strict=False)):
        box = FancyBboxPatch(
            (x_position - box_width / 2, y_position - box_height / 2),
            box_width,
            box_height,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1.4,
            edgecolor="#1f2937",
            facecolor=color,
        )
        axis.add_patch(box)
        axis.text(
            x_position,
            y_position,
            label,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="semibold",
        )
        if index < len(steps) - 1:
            arrow = FancyArrowPatch(
                (x_position + box_width / 2, y_position),
                (x_positions[index + 1] - box_width / 2, y_position),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.6,
                color="#495057",
            )
            axis.add_patch(arrow)

    axis.text(
        0.5,
        0.15,
        "Final deployment model: tuned TF-IDF. BERT and ensemble remain semantic comparison baselines.",
        ha="center",
        va="center",
        fontsize=10,
        color="#343a40",
    )
    axis.set_title("Journal Recommendation System Pipeline", fontsize=14, pad=12)
    return axis


def default_case_studies() -> list[dict[str, str]]:
    return [
        {
            "case_id": "case_ai_01",
            "topic": "Deep Learning / Artificial Intelligence",
            "abstract": (
                "This study proposes a transformer-based deep learning framework for medical image classification. "
                "The model combines attention-guided feature extraction, contrastive pretraining, and uncertainty-aware "
                "decision calibration to improve robustness across multi-center imaging datasets. Experimental results "
                "show higher classification accuracy and better generalization than conventional convolutional baselines."
            ),
        },
        {
            "case_id": "case_net_01",
            "topic": "Computer Networks / Wireless Systems",
            "abstract": (
                "We present an adaptive routing and congestion-control strategy for wireless sensor and edge networks. "
                "The method jointly optimizes packet scheduling, link-quality estimation, and energy-aware forwarding "
                "to reduce delay and packet loss under changing traffic conditions. Simulations demonstrate improved "
                "throughput and network lifetime relative to existing distributed routing protocols."
            ),
        },
        {
            "case_id": "case_se_01",
            "topic": "Software Engineering / Quality Assurance",
            "abstract": (
                "This paper introduces a data-driven defect prediction pipeline for large software repositories. "
                "Static code metrics, issue history, and change-level process features are fused to identify fault-prone "
                "modules before release. The empirical evaluation shows that the proposed model improves early bug detection "
                "and supports more effective testing prioritization in continuous integration environments."
            ),
        },
    ]


def build_case_study_examples(models: dict[str, Any], *, top_k: int = 5) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for case in default_case_studies():
        recommendations = {
            model_name: model.recommend(case["abstract"], top_k=top_k) for model_name, model in models.items()
        }
        payload.append(
            {
                **case,
                "final_model": "tfidf",
                "final_model_name": "TF-IDF Recommender",
                "recommendations": recommendations,
                "top_1_by_model": {
                    model_name: model_recommendations[0]["journal"] if model_recommendations else ""
                    for model_name, model_recommendations in recommendations.items()
                },
            }
        )
    return payload


def summarize_confidence_performance(
    predictions: pd.DataFrame,
    *,
    low_quantile: float = 0.25,
    high_quantile: float = 0.75,
    example_count: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "predicted_confidence" not in predictions.columns:
        raise KeyError("Predictions must include a predicted_confidence column.")

    frame = predictions.copy()
    low_threshold = float(frame["predicted_confidence"].quantile(low_quantile))
    high_threshold = float(frame["predicted_confidence"].quantile(high_quantile))
    frame["confidence_group"] = np.where(
        frame["predicted_confidence"] <= low_threshold,
        "Low confidence",
        np.where(frame["predicted_confidence"] >= high_threshold, "High confidence", "Mid confidence"),
    )

    summary = (
        frame.groupby("confidence_group")
        .agg(
            count=("record_id", "size"),
            top_1_accuracy=("is_top_1", "mean"),
            top_5_hit_rate=("is_top_5", "mean"),
            mean_confidence=("predicted_confidence", "mean"),
            median_margin=("score_margin", "median"),
            median_true_rank=("true_rank", "median"),
        )
        .reset_index()
    )
    summary["top_1_accuracy"] = summary["top_1_accuracy"].round(4)
    summary["top_5_hit_rate"] = summary["top_5_hit_rate"].round(4)
    summary["mean_confidence"] = summary["mean_confidence"].round(4)
    summary["median_margin"] = summary["median_margin"].round(4)
    summary["median_true_rank"] = summary["median_true_rank"].round(2)

    candidate_columns = [
        "record_id",
        "title",
        "true_journal",
        "predicted_journal",
        "runner_up_journal",
        "predicted_confidence",
        "score_margin",
        "true_rank",
        "is_top_1",
        "is_top_5",
        "cluster_label",
        "subjects",
        "abstract",
    ]
    selected_columns = [column for column in candidate_columns if column in frame.columns]
    high_examples = (
        frame.loc[frame["confidence_group"].eq("High confidence")]
        .sort_values(["predicted_confidence", "true_rank"], ascending=[False, True])
        .head(example_count)[selected_columns]
        .reset_index(drop=True)
    )
    low_examples = (
        frame.loc[frame["confidence_group"].eq("Low confidence")]
        .sort_values(["predicted_confidence", "true_rank"], ascending=[True, False])
        .head(example_count)[selected_columns]
        .reset_index(drop=True)
    )
    return summary, high_examples, low_examples


def plot_confidence_distribution(predictions: pd.DataFrame, *, ax: Axes | None = None) -> Axes:
    if "predicted_confidence" not in predictions.columns:
        raise KeyError("Predictions must include a predicted_confidence column.")

    axis = ax or plt.subplots(figsize=(10, 5))[1]
    bins = np.linspace(0, 1, 21)
    correct = predictions.loc[predictions["is_top_1"], "predicted_confidence"]
    incorrect = predictions.loc[~predictions["is_top_1"], "predicted_confidence"]
    axis.hist(correct, bins=bins, alpha=0.75, color="#2b8a3e", label="Correct Top-1 prediction")
    axis.hist(incorrect, bins=bins, alpha=0.65, color="#c92a2a", label="Incorrect Top-1 prediction")

    low_threshold = float(predictions["predicted_confidence"].quantile(0.25))
    high_threshold = float(predictions["predicted_confidence"].quantile(0.75))
    axis.axvline(low_threshold, color="#1c7ed6", linestyle="--", linewidth=1.2, label="Low-confidence quartile")
    axis.axvline(high_threshold, color="#5f3dc4", linestyle="--", linewidth=1.2, label="High-confidence quartile")
    axis.set_title("Confidence Distribution for TF-IDF Predictions")
    axis.set_xlabel("Predicted Confidence Score")
    axis.set_ylabel("Number of Test Articles")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    return axis
