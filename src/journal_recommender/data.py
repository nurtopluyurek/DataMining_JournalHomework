from __future__ import annotations

import html
import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .config import DEFAULT_DB_NAME, resolve_input_path


WORD_RE = re.compile(r"[^a-z0-9\s]+")
MULTISPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class DatasetConfig:
    db_path: Path | None = None
    profile: str = "assignment_aligned"
    target_journals: int = 175
    target_records: int = 7711
    allowed_document_types: tuple[str, ...] = ("Article", "Review")
    allowed_publication_types: tuple[str, ...] = ("Journal",)
    language: str = "English"

    def resolved_db_path(self) -> Path:
        return (self.db_path or resolve_input_path(DEFAULT_DB_NAME)).resolve()


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    resolved = (db_path or resolve_input_path(DEFAULT_DB_NAME)).resolve()
    connection = sqlite3.connect(resolved)
    connection.row_factory = sqlite3.Row
    return connection


def inspect_sqlite_schema(
    db_path: Path | None = None,
    sample_rows: int = 3,
) -> dict[str, Any]:
    connection = get_connection(db_path)
    cursor = connection.cursor()
    tables = [
        row["name"]
        for row in cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    ]
    report: dict[str, Any] = {"tables": tables, "table_details": {}}
    for table in tables:
        columns = cursor.execute(f"PRAGMA table_info('{table}')").fetchall()
        foreign_keys = cursor.execute(f"PRAGMA foreign_key_list('{table}')").fetchall()
        rows = cursor.execute(f"SELECT * FROM '{table}' LIMIT {int(sample_rows)}").fetchall()
        report["table_details"][table] = {
            "columns": [
                {
                    "cid": col["cid"],
                    "name": col["name"],
                    "type": col["type"],
                    "notnull": col["notnull"],
                    "default": col["dflt_value"],
                    "primary_key": bool(col["pk"]),
                }
                for col in columns
            ],
            "primary_keys": [col["name"] for col in columns if col["pk"]],
            "foreign_keys": [
                {"from": fk["from"], "table": fk["table"], "to": fk["to"]}
                for fk in foreign_keys
            ],
            "sample_rows": [dict(row) for row in rows],
        }
    connection.close()
    return report


def schema_summary_frame(schema_report: dict[str, Any]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for table_name, details in schema_report["table_details"].items():
        for column in details["columns"]:
            records.append(
                {
                    "table": table_name,
                    "column": column["name"],
                    "type": column["type"],
                    "is_primary_key": column["primary_key"],
                    "not_null": bool(column["notnull"]),
                }
            )
    return pd.DataFrame(records)


def strip_html(text: str | None) -> str:
    if text is None:
        return ""
    text = html.unescape(str(text)).replace("\x00", " ")
    text = BeautifulSoup(text, "html.parser").get_text(" ")
    return MULTISPACE_RE.sub(" ", text).strip()


def normalize_text(text: str | None) -> str:
    text = strip_html(text).lower()
    text = WORD_RE.sub(" ", text)
    tokens = [token for token in MULTISPACE_RE.sub(" ", text).strip().split() if token not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)


def load_raw_dataset(config: DatasetConfig | None = None) -> pd.DataFrame:
    config = config or DatasetConfig()
    connection = get_connection(config.resolved_db_path())
    query = """
    WITH keyword_map AS (
        SELECT
            ark.AcademicRecordId AS article_id,
            GROUP_CONCAT(DISTINCT ak.Name) AS keywords
        FROM AcademicRecordKeyword ark
        JOIN AcademicKeyword ak
            ON ark.AcademicKeywordId = ak.AcademicKeywordID
        GROUP BY ark.AcademicRecordId
    ),
    subject_map AS (
        SELECT
            ars.AcademicRecordId AS article_id,
            GROUP_CONCAT(DISTINCT s.NameEn) AS subjects
        FROM AcademicRecordSubject ars
        JOIN AcademicSubject s
            ON ars.AcademicSubjectId = s.AcademicSubjectID
        GROUP BY ars.AcademicRecordId
    ),
    keyword_plus_map AS (
        SELECT
            arkp.AcademicRecordId AS article_id,
            GROUP_CONCAT(DISTINCT kp.Name) AS keyword_plus
        FROM AcademicRecordKeywordPlus arkp
        JOIN AcademicKeywordPlus kp
            ON arkp.AcademicKeywordPlusId = kp.AcademicKeywordPlusID
        GROUP BY arkp.AcademicRecordId
    )
    SELECT
        ar.AcademicRecordID AS article_id,
        ar.WosUID AS wos_uid,
        ar.Title AS title,
        ara.AbstractText AS abstract_html,
        COALESCE(p.Name, 'Unknown Journal') AS journal_name,
        COALESCE(km.keywords, '') AS keywords,
        COALESCE(sm.subjects, '') AS subjects,
        COALESCE(kpm.keyword_plus, '') AS keyword_plus,
        ar.PubYear AS pub_year,
        ar.PubMonth AS pub_month,
        ar.PubDate AS pub_date,
        COALESCE(dt.NameEn, 'Unknown') AS document_type,
        COALESCE(pt.NameEn, 'Unknown') AS publication_type,
        COALESCE(l.Name, 'Unknown') AS language
    FROM AcademicRecord ar
    JOIN AcademicRecordAbstract ara
        ON ar.AcademicRecordID = ara.AcademicRecordId
    LEFT JOIN Publication p
        ON ar.PublicationId = p.PublicationID
    LEFT JOIN DocumentType dt
        ON ar.DocumentTypeId = dt.DocumentTypeID
    LEFT JOIN PublicationType pt
        ON ar.PublicationTypeId = pt.PublicationTypeID
    LEFT JOIN Language l
        ON ar.LanguageId = l.LanguageID
    LEFT JOIN keyword_map km
        ON ar.AcademicRecordID = km.article_id
    LEFT JOIN subject_map sm
        ON ar.AcademicRecordID = sm.article_id
    LEFT JOIN keyword_plus_map kpm
        ON ar.AcademicRecordID = kpm.article_id
    WHERE ara.AbstractText IS NOT NULL
    """
    dataset = pd.read_sql_query(query, connection)
    connection.close()
    dataset["abstract"] = dataset["abstract_html"].map(strip_html)
    dataset.drop(columns=["abstract_html"], inplace=True)
    dataset["title"] = dataset["title"].fillna("").astype(str).str.strip()
    dataset["journal_name"] = dataset["journal_name"].fillna("Unknown Journal").astype(str)
    dataset["keywords"] = dataset["keywords"].fillna("")
    dataset["subjects"] = dataset["subjects"].fillna("")
    dataset["keyword_plus"] = dataset["keyword_plus"].fillna("")
    dataset["pub_year"] = pd.to_numeric(dataset["pub_year"], errors="coerce").astype("Int64")
    dataset["pub_date"] = pd.to_datetime(dataset["pub_date"], errors="coerce")
    dataset["document_type"] = dataset["document_type"].fillna("Unknown")
    dataset["publication_type"] = dataset["publication_type"].fillna("Unknown")
    dataset["language"] = dataset["language"].fillna("Unknown")

    if config.allowed_document_types:
        dataset = dataset[dataset["document_type"].isin(config.allowed_document_types)]
    if config.allowed_publication_types:
        dataset = dataset[dataset["publication_type"].isin(config.allowed_publication_types)]
    if config.language:
        dataset = dataset[dataset["language"] == config.language]
    dataset = dataset[dataset["abstract"].str.len() > 0].copy()
    dataset.sort_values(["journal_name", "pub_year", "article_id"], inplace=True)
    dataset.reset_index(drop=True, inplace=True)
    return dataset


def _journal_stability_frame(dataset: pd.DataFrame) -> pd.DataFrame:
    yearly = (
        dataset.groupby(["journal_name", "pub_year"])["article_id"]
        .count()
        .rename("articles_in_year")
        .reset_index()
    )
    stats = (
        dataset.groupby("journal_name")
        .agg(
            article_count=("article_id", "count"),
            year_coverage=("pub_year", "nunique"),
            first_year=("pub_year", "min"),
            last_year=("pub_year", "max"),
        )
        .reset_index()
    )
    year_stats = (
        yearly.groupby("journal_name")["articles_in_year"]
        .agg(["mean", "std", "max", "min"])
        .reset_index()
        .rename(
            columns={
                "mean": "articles_per_year_mean",
                "std": "articles_per_year_std",
                "max": "articles_per_year_max",
                "min": "articles_per_year_min",
            }
        )
    )
    stats = stats.merge(year_stats, on="journal_name", how="left")
    stats["articles_per_year_std"] = stats["articles_per_year_std"].fillna(0.0)
    stats["consistency_score"] = 1 / (1 + stats["articles_per_year_std"])
    stats["stability_score"] = (
        stats["year_coverage"] * 100
        + stats["consistency_score"] * 10
        + stats["article_count"] / 100
    )
    stats.sort_values(
        ["stability_score", "article_count", "journal_name"],
        ascending=[False, False, True],
        inplace=True,
    )
    stats.reset_index(drop=True, inplace=True)
    return stats


def _round_robin_selection(group: pd.DataFrame, target_count: int) -> pd.DataFrame:
    if len(group) <= target_count:
        return group.copy()
    group = group.copy()
    group["_sort_year"] = group["pub_year"].fillna(-1)
    group.sort_values(
        ["_sort_year", "pub_date", "article_id"],
        ascending=[False, False, True],
        inplace=True,
    )
    buckets = [
        bucket.drop(columns=["_sort_year"]).copy()
        for _, bucket in group.groupby("_sort_year", sort=False)
    ]
    selected_frames: list[pd.DataFrame] = []
    picked = 0
    while picked < target_count and any(not bucket.empty for bucket in buckets):
        for index, bucket in enumerate(buckets):
            if bucket.empty or picked >= target_count:
                continue
            selected_frames.append(bucket.iloc[[0]])
            buckets[index] = bucket.iloc[1:].copy()
            picked += 1
    selected = pd.concat(selected_frames, ignore_index=True)
    return selected


def build_assignment_aligned_dataset(
    dataset: pd.DataFrame,
    target_journals: int = 175,
    target_records: int = 7711,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stats = _journal_stability_frame(dataset)
    selected_journals = stats.head(target_journals)["journal_name"].tolist()
    subset = dataset[dataset["journal_name"].isin(selected_journals)].copy()

    base_target = target_records // target_journals
    remainder = target_records % target_journals
    journal_targets = {journal: base_target for journal in selected_journals}
    for journal in selected_journals[:remainder]:
        journal_targets[journal] += 1

    selected_rows: list[pd.DataFrame] = []
    for journal, group in subset.groupby("journal_name", sort=False):
        selected_rows.append(_round_robin_selection(group, journal_targets[journal]))
    aligned = pd.concat(selected_rows, ignore_index=True)
    aligned.sort_values(["journal_name", "pub_year", "article_id"], inplace=True)
    aligned.reset_index(drop=True, inplace=True)

    journal_summary = stats[stats["journal_name"].isin(selected_journals)].copy()
    journal_summary["selected_records"] = journal_summary["journal_name"].map(journal_targets)
    journal_summary["raw_records"] = journal_summary["article_count"]
    return aligned, journal_summary


def add_preprocessed_columns(dataset: pd.DataFrame) -> pd.DataFrame:
    processed = dataset.copy()
    for column in ("title", "abstract", "keywords", "subjects", "keyword_plus"):
        processed[f"{column}_clean"] = processed[column].fillna("").map(normalize_text)
    processed["combined_text"] = (
        processed["abstract_clean"]
        + " "
        + processed["keywords_clean"]
        + " "
        + processed["subjects_clean"]
        + " "
        + processed["keyword_plus_clean"]
    ).map(lambda value: MULTISPACE_RE.sub(" ", value).strip())

    def split_terms(value: str) -> list[str]:
        if not value:
            return []
        parts = [part.strip() for part in value.split(",") if part.strip()]
        seen: set[str] = set()
        ordered: list[str] = []
        for part in parts:
            lowered = normalize_text(part)
            if lowered and lowered not in seen:
                ordered.append(lowered)
                seen.add(lowered)
        return ordered

    processed["keyword_terms"] = processed["keywords"].fillna("").map(split_terms)
    processed["subject_terms"] = processed["subjects"].fillna("").map(split_terms)
    processed["keyword_plus_terms"] = processed["keyword_plus"].fillna("").map(split_terms)
    return processed


def build_dataset(config: DatasetConfig | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    config = config or DatasetConfig()
    raw_dataset = load_raw_dataset(config)
    profile_metadata: dict[str, Any] = {
        "profile": config.profile,
        "raw_records": int(len(raw_dataset)),
        "raw_journals": int(raw_dataset["journal_name"].nunique()),
    }
    if config.profile == "full":
        dataset = raw_dataset.copy()
    elif config.profile == "assignment_aligned":
        dataset, journal_summary = build_assignment_aligned_dataset(
            raw_dataset,
            target_journals=config.target_journals,
            target_records=config.target_records,
        )
        profile_metadata["journal_selection"] = journal_summary.to_dict(orient="records")
        profile_metadata["selected_journals"] = int(dataset["journal_name"].nunique())
        profile_metadata["selected_records"] = int(len(dataset))
    else:
        raise ValueError(f"Unknown dataset profile: {config.profile}")

    dataset = add_preprocessed_columns(dataset)
    profile_metadata["final_records"] = int(len(dataset))
    profile_metadata["final_journals"] = int(dataset["journal_name"].nunique())
    profile_metadata["year_span"] = {
        "min": int(dataset["pub_year"].min()) if dataset["pub_year"].notna().any() else None,
        "max": int(dataset["pub_year"].max()) if dataset["pub_year"].notna().any() else None,
    }
    return dataset.reset_index(drop=True), profile_metadata
