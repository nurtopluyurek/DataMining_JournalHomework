# IEEE-Style Report Outline

## Title

Computer Science Journal Recommendation System Using TF-IDF, Sentence-BERT, Hybrid Ranking, and Topic Discovery

## Abstract

- Problem: recommend the most suitable journals for a new computer science abstract
- Data source: relational academic publication database
- Methods: TF-IDF, Sentence-BERT, hybrid ranking, ensemble scoring, KMeans, LDA
- Key outputs: Top-5 journal recommendations, explainability, evaluation metrics
- Main findings: summarize best model and clustering behavior

## 1. Introduction

- Motivation: journal selection is difficult for researchers
- Practical importance of journal recommendation
- Project objective and deliverables
- Research questions:
  - Can abstract-driven similarity recommend relevant journals?
  - Does semantic modeling improve over TF-IDF?
  - Do clusters and topics reveal meaningful subfields?

## 2. Related Work / Literature Review

- Content-based recommendation systems
- Academic venue recommendation
- TF-IDF and cosine similarity in document retrieval
- Transformer-based semantic retrieval
- Clustering and topic modeling in scientific corpora
- Gap addressed by this project

## 3. Dataset and Database Structure

- Source files and assignment context
- Raw SQLite schema inspection
- Relevant entities:
  - `AcademicRecord`
  - `AcademicRecordAbstract`
  - `Publication`
  - `AcademicRecordKeyword`
  - `AcademicRecordSubject`
  - `AcademicRecordKeywordPlus`
- Join strategy and inferred relationships
- Raw/full corpus vs assignment-aligned subset

## 4. Data Preparation

- Filtering rules
- Handling missing values
- Aggregating one-to-many keyword/subject relations
- Text preprocessing:
  - lowercase
  - punctuation removal
  - stopword removal
  - HTML cleaning
- Final feature field: `combined_text`

## 5. Methodology

### 5.1 TF-IDF Baseline

- Vectorization setup
- Cosine similarity
- Journal ranking from similar articles

### 5.2 Sentence-BERT Semantic Model

- Selected transformer model
- Embedding generation
- Semantic similarity scoring

### 5.3 Hybrid Ranking

- Formula:
  - `0.6 * similarity + 0.2 * keyword_overlap + 0.2 * subject_overlap`
- Why metadata overlap improves explainability

### 5.4 Ensemble Model

- Formula:
  - `0.5 * tfidf + 0.5 * bert`
- Expected tradeoff between lexical and semantic matching

### 5.5 Explainability Layer

- Supporting articles
- Overlapping keywords
- Overlapping subjects
- Cluster label
- Confidence scoring

## 6. Unsupervised Analysis

### 6.1 KMeans Clustering

- Search range: `K=2..80`
- Elbow method
- Silhouette score
- Cluster-size analysis
- Final `K` selection criteria:
  - silhouette
  - elbow point
  - interpretability
  - reasonable cluster sizes

### 6.2 Topic Modeling with LDA

- Topic count choice
- Top words per topic
- Comparison with KMeans clusters

## 7. Experimental Evaluation

- Evaluation design: hide journal, query with abstract, exclude self-match
- Metrics:
  - Top-1 accuracy
  - Top-3 hit rate
  - Top-5 hit rate
- Compared models:
  - TF-IDF
  - BERT
  - Hybrid
  - Ensemble

## 8. Results

- Dataset summary statistics
- Recommendation examples
- Clustering diagnostics
- Topic tables
- Evaluation comparison table
- Confidence score examples

## 9. Error Analysis

- Representative failed cases
- Real journal vs predicted journal
- Causes of failure:
  - overlapping subfields
  - insufficient metadata overlap
  - broad abstracts
  - journal ambiguity

## 10. Journal Insights

- Most frequent journals
- Most frequent keywords
- Subject distribution
- Journal-topic relationships
- Cluster composition by journal

## 11. Discussion

- Strengths of the hybrid approach
- When TF-IDF works well
- When BERT adds value
- Interpretation of the final cluster count
- Limitations:
  - assignment subset reconstruction from larger SQLite file
  - transformer download/cache requirement
  - metadata sparsity in some records

## 12. Conclusion

- Summary of contributions
- Best-performing model
- Practical usefulness for authors
- Future improvements:
  - cross-encoder reranking
  - journal scope descriptions
  - citation/network features
  - abstract + title + author intention prompts

## References

- IEEE citation format
- Include academic venue recommendation, semantic retrieval, clustering, and topic-modeling references
