from fastembed import TextEmbedding
import numpy as np

# --- Код из Q1 для получения query_embedding ---
query = 'I just discovered the course. Can I join now?'
model_name = 'jinaai/jina-embeddings-v2-small-en'
embedding_model = TextEmbedding(model_name=model_name, cache_dir="local_cache") # Добавил cache_dir для возможного ускорения

query_embedding_list = list(embedding_model.embed([query]))
query_embedding = None
if query_embedding_list:
    query_embedding = query_embedding_list[0]
else:
    print("Failed to generate query embedding for Q1.")
    exit()
# --- Конец кода из Q1 ---


# --- Код для Q2 ---
doc_text_q2 = 'Can I still join the course after the start date?'

# Получение эмбеддинга для документа
doc_embedding_list_q2 = list(embedding_model.embed([doc_text_q2]))

if doc_embedding_list_q2:
    doc_embedding_q2 = doc_embedding_list_q2[0]

    # Рассчет косинусного сходства
    # Так как векторы нормализованы, dot product = cosine similarity
    cosine_similarity_q2 = np.dot(query_embedding, doc_embedding_q2)
    print(f"Cosine similarity for Q2: {cosine_similarity_q2}")
else:
    print("Failed to generate document embedding for Q2.")