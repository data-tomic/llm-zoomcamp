import requests
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
import numpy as np
import uuid # Для генерации уникальных ID, если не хотим использовать простые целые числа

# --- Конфигурация ---
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "ml_zoomcamp_faq"
MODEL_NAME_Q6 = 'BAAI/bge-small-en' # Модель из Q5
EMBEDDING_DIMENSIONALITY_Q6 = 384 # Размерность модели из Q5
QUERY_Q1 = 'I just discovered the course. Can I join now?'

# --- Инициализация клиентов ---
try:
    client = QdrantClient(url=QDRANT_URL)
except Exception as e:
    print(f"Failed to connect to Qdrant: {e}")
    print("Make sure Qdrant is running, e.g., in Docker.")
    exit()

embedding_model_q6 = TextEmbedding(model_name=MODEL_NAME_Q6, cache_dir="local_cache")

# --- 1. Загрузка и фильтрация документов ---
docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
try:
    docs_response = requests.get(docs_url)
    docs_response.raise_for_status() # Проверка на ошибки HTTP
    documents_raw = docs_response.json()
except requests.exceptions.RequestException as e:
    print(f"Failed to download documents: {e}")
    exit()

ml_documents = []
for course_data in documents_raw:
    if course_data.get('course') == 'machine-learning-zoomcamp':
        for doc in course_data.get('documents', []):
            # Добавляем поле 'course' внутрь каждого документа для удобства, если оно еще не там
            doc['course'] = course_data.get('course')
            ml_documents.append(doc)

print(f"Found {len(ml_documents)} documents for 'machine-learning-zoomcamp'.")

if not ml_documents:
    print("No documents found for machine-learning-zoomcamp. Exiting.")
    exit()

# --- 2. Подготовка и создание/пересоздание коллекции в Qdrant ---
# Попробуем удалить коллекцию, если она существует, чтобы начать с чистого листа
try:
    client.delete_collection(collection_name=COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' deleted successfully.")
except Exception as e:
    # Если коллекция не существует, это нормально
    if "not found" in str(e).lower() or "status_code=404" in str(e).lower():
         print(f"Collection '{COLLECTION_NAME}' does not exist, will create a new one.")
    else:
        print(f"Warning: Could not delete collection '{COLLECTION_NAME}': {e}")


try:
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=EMBEDDING_DIMENSIONALITY_Q6,
            distance=models.Distance.COSINE  # BGE модели обычно используют косинусное сходство
        )
    )
    print(f"Collection '{COLLECTION_NAME}' created successfully.")
except Exception as e:
    if "already exists" in str(e).lower():
        print(f"Collection '{COLLECTION_NAME}' already exists.")
    else:
        print(f"Failed to create collection '{COLLECTION_NAME}': {e}")
        exit()

# --- 3. Создание эмбеддингов и добавление точек в Qdrant ---
points_to_upsert = []
for i, doc in enumerate(ml_documents):
    combined_text = doc.get('question', '') + ' ' + doc.get('text', '')
    
    # Получаем эмбеддинг
    # .embed() возвращает генератор, поэтому берем первый элемент
    text_embedding_list = list(embedding_model_q6.embed([combined_text]))
    
    if text_embedding_list and isinstance(text_embedding_list[0], np.ndarray):
        text_embedding = text_embedding_list[0]
        
        # Сохраняем все поля документа в payload
        payload = doc.copy() # Копируем, чтобы не изменять оригинальный ml_documents
        
        points_to_upsert.append(models.PointStruct(
            id=i, # Можно использовать uuid.uuid4().hex для большей уникальности, если нужно
            vector=text_embedding.tolist(), # Qdrant ожидает список float
            payload=payload
        ))
    else:
        print(f"Warning: Could not generate embedding for document index {i}, skipping.")

if points_to_upsert:
    try:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points_to_upsert,
            wait=True # Ожидать завершения операции
        )
        print(f"Successfully upserted {len(points_to_upsert)} points into '{COLLECTION_NAME}'.")
    except Exception as e:
        print(f"Failed to upsert points: {e}")
        exit()
else:
    print("No points to upsert. Exiting.")
    exit()

# --- 4. Выполнение запроса ---
# Получаем эмбеддинг для запроса Q1
query_embedding_q1_list = list(embedding_model_q6.embed([QUERY_Q1]))

if not query_embedding_q1_list or not isinstance(query_embedding_q1_list[0], np.ndarray):
    print("Failed to generate embedding for the query from Q1.")
    exit()
    
query_vector_q1 = query_embedding_q1_list[0].tolist()

try:
    search_results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector_q1,
        limit=5 # Возьмем несколько результатов для просмотра, но для ответа нужен только первый
    )

    if search_results:
        highest_score = search_results[0].score
        print(f"\nSearch results for query: '{QUERY_Q1}'")
        # for i, hit in enumerate(search_results):
        #     print(f"Result {i+1}: Score: {hit.score:.4f}, Payload: {hit.payload.get('question')[:80]}...")
        print(f"\nHighest score in the results for Q6: {highest_score}")
    else:
        print("No results found for the query.")

except Exception as e:
    print(f"An error occurred during search: {e}")