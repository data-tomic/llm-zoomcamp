from fastembed import TextEmbedding
import numpy as np

# Текст запроса
query = 'I just discovered the course. Can I join now?'

# Имя модели
model_name = 'jinaai/jina-embeddings-v2-small-en'

# Создание объекта модели эмбеддингов
embedding_model = TextEmbedding(model_name=model_name)

# Получение эмбеддинга для запроса
# embed() возвращает генератор, поэтому преобразуем в список и берем первый элемент
query_embedding_list = list(embedding_model.embed([query]))

if query_embedding_list:
    query_embedding = query_embedding_list[0]
    print(f"Embedding shape: {query_embedding.shape}")

    # Нахождение минимального значения в эмбеддинге
    min_value = np.min(query_embedding)
    print(f"Minimal value in the embedding: {min_value}")

    # Для дополнительной проверки (не обязательно для ответа на вопрос)
    max_value = np.max(query_embedding)
    norm_value = np.linalg.norm(query_embedding)
    print(f"Maximal value in the embedding: {max_value}")
    print(f"Norm of the embedding: {norm_value}")
else:
    print("Failed to generate embedding.")