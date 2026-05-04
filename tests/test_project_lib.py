from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from project_lib import (
    EnsembleJournalRecommender,
    JournalRecommender,
    SemanticJournalRecommender,
    TopicClusterer,
    clean_text,
    filter_modeling_dataset,
)


def make_mock_frame() -> pd.DataFrame:
    rows = [
        {
            "record_id": 1,
            "title": "Deep learning for image recognition",
            "abstract": "Convolutional neural networks improve image classification.",
            "journal": "Journal A",
            "pub_year": 2018,
            "subjects": "Computer Science; Artificial Intelligence",
            "keywords": "deep learning; cnn",
            "keyword_plus": "image classification",
            "abstract_only_text": "convolutional neural networks improve image classification",
            "abstract_keywords_text": "convolutional neural networks improve image classification keywords deep learning cnn",
            "abstract_keywords_subjects_text": "convolutional neural networks improve image classification keywords deep learning cnn subjects computer science artificial intelligence",
            "combined_text": "deep learning image recognition convolutional neural networks improve image classification keywords deep learning cnn keyword plus image classification",
            "title_abstract_text": "deep learning image recognition convolutional neural networks improve image classification",
            "abstract_query_text": "convolutional neural networks improve image classification",
            "document_type": "Article",
            "publication_type": "Journal",
            "journal_article_count": 3,
            "subject_list": ["Computer Science", "Artificial Intelligence"],
            "keyword_list": ["deep learning", "cnn"],
            "keyword_plus_list": ["image classification"],
        },
        {
            "record_id": 2,
            "title": "Object detection with neural models",
            "abstract": "Neural models support robust object detection.",
            "journal": "Journal A",
            "pub_year": 2018,
            "subjects": "Computer Science; Artificial Intelligence",
            "keywords": "object detection; neural network",
            "keyword_plus": "computer vision",
            "abstract_only_text": "neural models support robust object detection",
            "abstract_keywords_text": "neural models support robust object detection keywords object detection neural network",
            "abstract_keywords_subjects_text": "neural models support robust object detection keywords object detection neural network subjects computer science artificial intelligence",
            "combined_text": "object detection with neural models neural models support robust object detection keywords object detection neural network keyword plus computer vision",
            "title_abstract_text": "object detection with neural models neural models support robust object detection",
            "abstract_query_text": "neural models support robust object detection",
            "document_type": "Article",
            "publication_type": "Journal",
            "journal_article_count": 3,
            "subject_list": ["Computer Science", "Artificial Intelligence"],
            "keyword_list": ["object detection", "neural network"],
            "keyword_plus_list": ["computer vision"],
        },
        {
            "record_id": 3,
            "title": "Survey of machine vision systems",
            "abstract": "Machine vision systems are widely used in pattern recognition.",
            "journal": "Journal A",
            "pub_year": 2019,
            "subjects": "Computer Science; Artificial Intelligence",
            "keywords": "machine vision; recognition",
            "keyword_plus": "pattern recognition",
            "abstract_only_text": "machine vision systems are widely used in pattern recognition",
            "abstract_keywords_text": "machine vision systems are widely used in pattern recognition keywords machine vision recognition",
            "abstract_keywords_subjects_text": "machine vision systems are widely used in pattern recognition keywords machine vision recognition subjects computer science artificial intelligence",
            "combined_text": "survey of machine vision systems machine vision systems are widely used in pattern recognition keywords machine vision recognition keyword plus pattern recognition",
            "title_abstract_text": "survey of machine vision systems machine vision systems are widely used in pattern recognition",
            "abstract_query_text": "machine vision systems are widely used in pattern recognition",
            "document_type": "Article",
            "publication_type": "Journal",
            "journal_article_count": 3,
            "subject_list": ["Computer Science", "Artificial Intelligence"],
            "keyword_list": ["machine vision", "recognition"],
            "keyword_plus_list": ["pattern recognition"],
        },
        {
            "record_id": 4,
            "title": "Database indexing for relational systems",
            "abstract": "Efficient indexing accelerates relational query execution.",
            "journal": "Journal B",
            "pub_year": 2018,
            "subjects": "Computer Science; Information Systems",
            "keywords": "database; indexing",
            "keyword_plus": "query optimization",
            "abstract_only_text": "efficient indexing accelerates relational query execution",
            "abstract_keywords_text": "efficient indexing accelerates relational query execution keywords database indexing",
            "abstract_keywords_subjects_text": "efficient indexing accelerates relational query execution keywords database indexing subjects computer science information systems",
            "combined_text": "database indexing for relational systems efficient indexing accelerates relational query execution keywords database indexing keyword plus query optimization",
            "title_abstract_text": "database indexing for relational systems efficient indexing accelerates relational query execution",
            "abstract_query_text": "efficient indexing accelerates relational query execution",
            "document_type": "Article",
            "publication_type": "Journal",
            "journal_article_count": 3,
            "subject_list": ["Computer Science", "Information Systems"],
            "keyword_list": ["database", "indexing"],
            "keyword_plus_list": ["query optimization"],
        },
        {
            "record_id": 5,
            "title": "Distributed query optimization techniques",
            "abstract": "Distributed query optimization reduces response time.",
            "journal": "Journal B",
            "pub_year": 2019,
            "subjects": "Computer Science; Information Systems",
            "keywords": "distributed database; optimization",
            "keyword_plus": "query planning",
            "abstract_only_text": "distributed query optimization reduces response time",
            "abstract_keywords_text": "distributed query optimization reduces response time keywords distributed database optimization",
            "abstract_keywords_subjects_text": "distributed query optimization reduces response time keywords distributed database optimization subjects computer science information systems",
            "combined_text": "distributed query optimization techniques distributed query optimization reduces response time keywords distributed database optimization keyword plus query planning",
            "title_abstract_text": "distributed query optimization techniques distributed query optimization reduces response time",
            "abstract_query_text": "distributed query optimization reduces response time",
            "document_type": "Article",
            "publication_type": "Journal",
            "journal_article_count": 3,
            "subject_list": ["Computer Science", "Information Systems"],
            "keyword_list": ["distributed database", "optimization"],
            "keyword_plus_list": ["query planning"],
        },
        {
            "record_id": 6,
            "title": "Transactional consistency in data systems",
            "abstract": "Consistency models are critical in distributed data systems.",
            "journal": "Journal B",
            "pub_year": 2019,
            "subjects": "Computer Science; Information Systems",
            "keywords": "transactions; consistency",
            "keyword_plus": "data systems",
            "abstract_only_text": "consistency models are critical in distributed data systems",
            "abstract_keywords_text": "consistency models are critical in distributed data systems keywords transactions consistency",
            "abstract_keywords_subjects_text": "consistency models are critical in distributed data systems keywords transactions consistency subjects computer science information systems",
            "combined_text": "transactional consistency in data systems consistency models are critical in distributed data systems keywords transactions consistency keyword plus data systems",
            "title_abstract_text": "transactional consistency in data systems consistency models are critical in distributed data systems",
            "abstract_query_text": "consistency models are critical in distributed data systems",
            "document_type": "Article",
            "publication_type": "Journal",
            "journal_article_count": 3,
            "subject_list": ["Computer Science", "Information Systems"],
            "keyword_list": ["transactions", "consistency"],
            "keyword_plus_list": ["data systems"],
        },
    ]
    return pd.DataFrame(rows)


class FakeSemanticJournalRecommender(SemanticJournalRecommender):
    def _load_model(self):
        return None

    def _encode_texts(self, texts, *, cache_path=None):
        embeddings = []
        for text in texts:
            value = text.lower()
            image_terms = sum(token in value for token in ["image", "vision", "recognition", "neural"])
            db_terms = sum(token in value for token in ["query", "database", "data", "index", "distributed"])
            embeddings.append([float(image_terms), float(db_terms), float(len(value.split()))])
        matrix = pd.DataFrame(embeddings, columns=["a", "b", "c"]).to_numpy(dtype="float32")
        norms = (matrix**2).sum(axis=1, keepdims=True) ** 0.5
        return matrix / norms


class ProjectLibTests(unittest.TestCase):
    def test_clean_text_handles_missing_values(self) -> None:
        self.assertEqual(clean_text(None), "")
        self.assertEqual(clean_text(float("nan")), "")
        self.assertEqual(clean_text("<p>Hello, World!</p>"), "hello world")

    def test_filter_modeling_dataset_excludes_long_tail_journals(self) -> None:
        frame = make_mock_frame()
        extra = frame.iloc[[0]].copy()
        extra["journal"] = "Journal C"
        frame = pd.concat([frame, extra], ignore_index=True)
        filtered, stats = filter_modeling_dataset(frame, min_journal_frequency=2)
        self.assertNotIn("Journal C", filtered["journal"].unique())
        self.assertEqual(stats["excluded_long_tail_journals"], 1)

    def test_recommender_smoke(self) -> None:
        frame = make_mock_frame()
        train = frame.iloc[[0, 1, 3, 4]].reset_index(drop=True)
        test = frame.iloc[[2, 5]].reset_index(drop=True)

        model = JournalRecommender(
            text_column="title_abstract_text",
            classifier_weight=1.0,
            similarity_weight=0.0,
            candidate_top_n=None,
            classifier_c=0.5,
            sublinear_tf=True,
            min_df=1,
            max_features=1000,
        )
        model.fit(train)
        recommendations = model.recommend("distributed query optimization reduces response time", top_k=2)
        self.assertEqual(len(recommendations), 2)

        result = model.evaluate(test, query_column="abstract_query_text", name="smoke")
        self.assertIn("top_5_accuracy", result.metrics)
        self.assertEqual(len(result.predictions), len(test))

    def test_clusterer_smoke(self) -> None:
        frame = make_mock_frame()
        clusterer = TopicClusterer(
            text_column="title_abstract_text",
            candidate_clusters=(2, 3),
            svd_components=2,
            max_features=500,
            min_df=1,
            silhouette_sample_size=6,
        )
        clusterer.fit(frame)
        summary = clusterer.summarize_clusters()
        self.assertFalse(summary.empty)
        self.assertIn("top_terms", summary.columns)

    def test_semantic_recommender_smoke(self) -> None:
        frame = make_mock_frame()
        train = frame.iloc[[0, 1, 3, 4]].reset_index(drop=True)
        test = frame.iloc[[2, 5]].reset_index(drop=True)

        model = FakeSemanticJournalRecommender(text_column="combined_text", n_neighbors=3)
        model.fit(train)
        recommendations = model.recommend_journals_bert("machine vision and recognition", top_n=2)
        self.assertEqual(len(recommendations), 2)
        self.assertIn("confidence_score", recommendations[0])
        self.assertIn("explanation", recommendations[0])

        result = model.evaluate(test, query_column="abstract_query_text", name="semantic")
        self.assertIn("top_5_accuracy", result.metrics)
        self.assertEqual(len(result.predictions), len(test))

    def test_ensemble_recommender_smoke(self) -> None:
        frame = make_mock_frame()
        train = frame.iloc[[0, 1, 3, 4]].reset_index(drop=True)
        test = frame.iloc[[2, 5]].reset_index(drop=True)

        tfidf_model = JournalRecommender(
            text_column="title_abstract_text",
            classifier_weight=1.0,
            similarity_weight=0.0,
            candidate_top_n=None,
            classifier_c=0.5,
            sublinear_tf=True,
            min_df=1,
            max_features=1000,
        )
        bert_model = FakeSemanticJournalRecommender(text_column="combined_text", n_neighbors=3)
        ensemble = EnsembleJournalRecommender(tfidf_model, bert_model, tfidf_weight=0.5, bert_weight=0.5)
        ensemble.fit(train)
        recommendations = ensemble.recommend_journals_ensemble("distributed query optimization", top_n=2)
        self.assertEqual(len(recommendations), 2)
        self.assertIn("bert_score", recommendations[0])
        self.assertIn("tfidf_score", recommendations[0])

        result = ensemble.evaluate(test, query_column="abstract_query_text", name="ensemble")
        self.assertIn("macro_f1", result.metrics)


if __name__ == "__main__":
    unittest.main()
