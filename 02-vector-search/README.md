# LLM Zoomcamp - Vector Search Homework

This document outlines the steps and findings for the Vector Search homework assignment in the LLM Zoomcamp. The primary goal was to explore text embeddings, cosine similarity, and the use of Qdrant with fastembed for vector search tasks.

## Tools Used

*   Python 3.x
*   `fastembed` library for generating text embeddings.
*   `numpy` for numerical operations, especially on embeddings.
*   `qdrant-client` for interacting with the Qdrant vector database.
*   `requests` for fetching data.
*   A local Qdrant instance running in Docker.

## Homework Questions & Process

Below is a summary of each question, the process undertaken, and the results obtained.

### Q1. Embedding the query

**Task:** Embed the query: 'I just discovered the course. Can I join now?' using the 'jinaai/jina-embeddings-v2-small-en' model and find the minimal value in the resulting 512-dimension numpy array.

**Process:**
1.  Initialized `TextEmbedding` with `model_name='jinaai/jina-embeddings-v2-small-en'`.
2.  Embedded the query string.
3.  Used `numpy.min()` to find the minimal value in the embedding vector.

**Result:**
*   Minimal value obtained: `-0.11726373885183883`
*   Closest answer from options: **-0.11**

### Q2. Cosine similarity with another vector

**Task:** Embed the document: 'Can I still join the course after the start date?' and compute the cosine similarity between its vector and the query vector from Q1.

**Process:**
1.  Embedded the document string using the same model.
2.  Calculated the dot product between the normalized query embedding (from Q1) and the normalized document embedding. Since `fastembed` models often return normalized vectors, the dot product is equivalent to cosine similarity.

**Result:**
*   Cosine similarity obtained: `0.9008528895674548`
*   Closest answer from options: **0.9**

### Q3. Ranking by cosine

**Task:** For a given list of 5 documents, compute embeddings for their 'text' field and find the document index with the highest cosine similarity to the Q1 query vector.

**Process:**
1.  Iterated through the provided documents, extracting the 'text' field.
2.  Embedded each 'text' field.
3.  Calculated the cosine similarity (dot product) between the Q1 query embedding and each document embedding.
4.  Used `numpy.argmax()` to find the index of the document with the highest similarity score.

**Result:**
*   Similarities: `[0.7629684696540238, 0.8182378150042889, 0.8085397398734189, 0.7133079015686243, 0.7304499234333598]`
*   Index of document with highest similarity: `1`
*   Chosen answer: **1**

### Q4. Ranking by cosine, version two

**Task:** For the same 5 documents, create a `full_text` field by concatenating `doc['question'] + ' ' + doc['text']`. Embed this `full_text` and find the document with the highest cosine similarity to the Q1 query vector.

**Process:**
1.  For each document, created the `full_text` string.
2.  Embedded each `full_text` string.
3.  Calculated the cosine similarity (dot product) between the Q1 query embedding and each `full_text` embedding.
4.  Used `numpy.argmax()` to find the index of the document with the highest similarity score.

**Result:**
*   Similarities (full_text): `[0.8514543236908068, 0.8436594159113068, 0.8408287048502558, 0.7755157969663908, 0.8086007795043938]`
*   Index of document with highest similarity (full_text): `0`
*   Chosen answer: **0**

**Observation:** The highest-scoring document changed from index 1 (Q3) to index 0 (Q4). This is likely because concatenating the question and answer provides a more comprehensive semantic context for the query, making document 0 (whose question was 'Course - Can I still join the course after the start date?') a better overall match to the query 'I just discovered the course. Can I join now?'.

### Q5. Selecting the embedding model

**Task:** Determine the smallest dimensionality among models available in `fastembed`.

**Process:**
1.  Used `TextEmbedding.list_supported_models()` to get a list of all supported models.
2.  Iterated through the list, examining the 'dim' (dimensionality) field for each model.
3.  Found the minimum dimensionality value.

**Result:**
*   Smallest dimensionality found: `384`
*   Chosen answer: **384**
    *   One such model identified was `BAAI/bge-small-en`.

### Q6. Indexing with qdrant

**Task:**
1.  Filter documents from the full dataset for `course == 'machine-learning-zoomcamp'`.
2.  For these documents, create a combined text field: `text = doc['question'] + ' ' + doc['text']`.
3.  Add these documents to a Qdrant collection using the `BAAI/bge-small-en` model (dimensionality 384).
4.  Query this collection using the Q1 query and find the highest score.

**Process:**
1.  Fetched the full `documents.json` dataset.
2.  Filtered documents for the 'machine-learning-zoomcamp' course.
3.  Initialized `QdrantClient` and `TextEmbedding` with `model_name='BAAI/bge-small-en'`.
4.  Created/recreated a Qdrant collection named `ml_zoomcamp_faq` with vector size 384 and COSINE distance.
5.  For each filtered document, concatenated its 'question' and 'text' fields.
6.  Embedded the concatenated string using `BAAI/bge-small-en`.
7.  Upserted points into Qdrant, with the embedding as the vector and the original document fields as payload.
8.  Embedded the Q1 query string using the same `BAAI/bge-small-en` model.
9.  Searched the Qdrant collection using the query vector.
10. Extracted the highest score from the search results.

**Result:**
*   Number of 'machine-learning-zoomcamp' documents found: `375`
*   Highest score obtained from Qdrant search: `0.8703172`
*   Closest answer from options: **0.87**

## Setup and Execution Notes

*   A Python virtual environment (`llm_env`) was created and activated.
*   Required libraries (`fastembed`, `numpy`, `requests`, `qdrant-client`) were installed using `pip`.
*   Qdrant was run as a Docker container on `http://localhost:6333`.
*   Python scripts were executed locally within the activated virtual environment to obtain the answers.