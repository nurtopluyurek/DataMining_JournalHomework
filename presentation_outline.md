# Presentation Outline

## Slide 1 — Title

- Computer Science Journal Recommendation System
- Course, student name, date

Speaker notes:
This project recommends the top 5 most relevant journals for a new computer science abstract using both lexical and semantic data mining methods.

## Slide 2 — Problem and Motivation

- Researchers struggle to choose the best-fit journal
- Poor journal targeting wastes submission time
- Need a data-driven recommendation system

Speaker notes:
The idea is to use past journal articles and their metadata to estimate which journals are most relevant for a new manuscript abstract.

## Slide 3 — Dataset and Database

- Input files: SQLite database, archive, assignment PDF
- Key entities: articles, abstracts, publications, keywords, subjects, keyword-plus
- Schema inspected dynamically with SQL

Speaker notes:
I did not assume table names blindly. The first step was schema inspection using `sqlite_master` and `PRAGMA table_info`, then I identified the actual join path between articles, abstracts, and journals.

## Slide 4 — Data Preparation

- Filtered to English journal articles/reviews with abstracts
- Aggregated one-to-many keywords and subjects
- Cleaned HTML and normalized text
- Built `combined_text`

Speaker notes:
The final modeling table includes the article ID, title, abstract, journal name, keywords, subjects, and keyword-plus terms. One-to-many relations were grouped into strings and reusable term lists.

## Slide 5 — Recommendation Models

- TF-IDF baseline
- Sentence-BERT semantic model
- Hybrid ranking
- Ensemble model

Speaker notes:
TF-IDF captures exact lexical similarity, BERT captures semantic similarity, the hybrid model adds keyword and subject overlap, and the ensemble balances lexical and semantic evidence.

## Slide 6 — Explainability

- Supporting similar articles
- Overlapping keywords
- Overlapping subjects
- Cluster label
- Confidence score

Speaker notes:
The system is not just a black box. Each recommendation explains why the journal was suggested using interpretable evidence from the corpus.

## Slide 7 — Clustering Analysis

- KMeans with `K=2..80`
- Elbow + silhouette
- Cluster-size balance
- Final cluster count rationale

Speaker notes:
I did not choose the maximum silhouette blindly. I balanced the curve behavior with interpretability and avoided solutions that create too many tiny clusters.

## Slide 8 — Topic Modeling

- LDA on the same corpus
- Top words per topic
- Comparison with KMeans clusters

Speaker notes:
KMeans groups documents by vector similarity, while LDA tries to discover latent semantic themes. Together they help interpret the journal landscape.

## Slide 9 — Evaluation

- Hidden-journal testing
- Top-1 accuracy
- Top-3 hit rate
- Top-5 hit rate
- Model comparison

Speaker notes:
For every article, I used the abstract as input and excluded the article itself from retrieval. Then I checked whether the true journal appeared in the top 1, top 3, or top 5 results.

## Slide 10 — Error Analysis

- Typical failure cases
- Real journal vs predicted journal
- Why errors happen

Speaker notes:
Most errors come from overlapping subfields, broad abstracts, or journals that publish very similar topics. These are realistic ambiguity cases, not random failures.

## Slide 11 — Journal Insights and Demo

- Most frequent journals
- Keyword distribution
- Subject distribution
- Streamlit demo screenshot / live demo

Speaker notes:
Beyond recommendation, the project also provides descriptive insights into the journal corpus and supports a practical interface for testing new abstracts.

## Slide 12 — Conclusion and Future Work

- Hybrid model is usually the best tradeoff
- Explainability improves usability
- Future work: reranking, citation features, richer metadata

Speaker notes:
The final system combines database engineering, text mining, semantic retrieval, clustering, and evaluation into a single reproducible project that can be extended further.
