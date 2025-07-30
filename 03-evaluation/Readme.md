# LLM Zoomcamp - Module 3 Homework: Evaluating RAG Systems

This project explores various methods for evaluating the retrieval and generation components of a Retrieval-Augmented Generation (RAG) system. The goal is to compare different search techniques, from basic keyword search to modern vector databases with advanced sentence embeddings, and to evaluate the quality of the final generated answers.

The evaluation is performed on the FAQ dataset from the DataTalks.Club Machine Learning Zoomcamp.

## Project Setup

### 1. Prerequisites
- Python 3.10+
- `pip` for package management

### 2. Create a Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies.

```bash
# Create the virtual environment
python -m venv venv

# Activate the environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
.\venv\Scripts\activate
```

### 3. Install Dependencies
All required Python libraries can be installed with the following command:

```bash
pip install -U minsearch qdrant_client pandas scikit-learn requests tqdm rouge numpy sentence-transformers
```

## How to Run
The entire workflow, including data download, indexing, and evaluation for all homework questions, is contained in the main script.

To run the evaluation, simply execute the script:
```bash
python homework.py
```
The script will print the results for each question to the console. The first run may take a few minutes as it needs to download the datasets and the sentence transformer model.

## Results Summary

The following table summarizes the results obtained for each question in the homework.

| Question | Method | Metric | Result |
| :--- | :--- | :--- | :--- |
| **Q1** | Keyword Search (`minsearch`) with boosting | Hit-Rate | `0.849` |
| **Q2** | Vector Search (TF-IDF+SVD on Questions only) | MRR | `0.357` |
| **Q3** | Vector Search (TF-IDF+SVD on Q+A) | Hit-Rate | `0.821` |
| **Q4** | Vector Search (`JinaAI` + `Qdrant`) | MRR | `0.831` |
| **Q5** | Generation Eval (TF-IDF+SVD) | Avg. Cosine Similarity | `0.842` |
| **Q6** | Generation Eval (`rouge` library) | Avg. ROUGE-1 F1 | `0.352` |

---

## Detailed Conclusions

This homework provided a comprehensive overview of evaluation techniques for RAG pipelines, yielding several key insights.

### 1. Keyword Search Remains a Strong Baseline
The initial evaluation using `minsearch` with boosted fields (Q1) achieved a **Hit-Rate of ~0.85**. This demonstrates that a well-configured lexical search system can be surprisingly effective, especially when the query phrasing is similar to the questions in the knowledge base. It serves as a robust baseline that more complex systems must outperform.

### 2. The Importance of Data Quality in Embeddings
The sharp decline in performance in Q2 (MRR `~0.36`) was a critical lesson. By creating embeddings from questions alone, we discarded the rich context available in the answers. The system's performance was restored and significantly improved in Q3 (Hit-Rate `~0.82`) by creating embeddings from the combined question and answer text. This highlights a fundamental principle: the quality of a semantic search system is directly proportional to the quality and completeness of the data fed into the embedding model.

### 3. Modern Embeddings and Vector Databases Show Superior Performance
Transitioning to a modern sentence transformer model (`jinaai/jina-embeddings-v2-small-en`) and a dedicated vector database (`Qdrant`) in Q4 resulted in the best retrieval performance, with an **MRR of ~0.83**. This approach outperformed the TF-IDF/SVD method, proving the superior ability of deep learning models to capture true semantic meaning.

A crucial finding was the necessity of using model-specific prefixes (`search_query:` and `search_document:`). Without them, performance was extremely poor (MRR `~0.13`), but with them, it became the top-performing retrieval strategy. This underscores the importance of carefully reading the documentation for any pre-trained model to ensure it is used as intended.

### 4. Differentiating Semantic and Lexical Similarity in Generation
The final two questions evaluated the generated text itself. We observed a **high average cosine similarity (`~0.84`)** but a **low average ROUGE-1 F1 score (`~0.35`)**. This is not a contradiction but a valuable insight.
- The **high cosine similarity** indicates that the LLM-generated answers were semantically aligned with the ground-truth answers. The core meaning and intent were correctly captured.
- The **low ROUGE score** indicates that the answers were not lexically identical. The LLM rephrased the original answers, used synonyms, and altered sentence structures, as a human would.

This combination is often the signature of a high-quality RAG system: it provides answers that are factually correct and contextually relevant, but articulated in a natural, non-repetitive manner.
