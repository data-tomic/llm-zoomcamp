# 1. Импорт библиотек
import minsearch
import requests
import pandas as pd
from tqdm.auto import tqdm

# 2. Загрузка данных
# Убедитесь, что у вас есть доступ в интернет для скачивания файлов
print("Загрузка данных...")
url_prefix = 'https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/03-evaluation/'
docs_url = url_prefix + 'search_evaluation/documents-with-ids.json'
documents = requests.get(docs_url).json()

ground_truth_url = url_prefix + 'search_evaluation/ground-truth-data.csv'
df_ground_truth = pd.read_csv(ground_truth_url)
ground_truth = df_ground_truth.to_dict(orient='records')
print("Данные успешно загружены.")

# 3. Функции для оценки (из текста задания)
def hit_rate(relevance_total):
    cnt = 0
    for line in relevance_total:
        if True in line:
            cnt = cnt + 1
    return cnt / len(relevance_total)

def mrr(relevance_total):
    total_score = 0.0
    for line in relevance_total:
        for rank in range(len(line)):
            if line[rank] == True:
                total_score = total_score + 1 / (rank + 1)
    return total_score / len(relevance_total)

def evaluate(ground_truth, search_function):
    relevance_total = []
    for q in tqdm(ground_truth):
        doc_id = q['document']
        results = search_function(q)
        relevance = [d['id'] == doc_id for d in results]
        relevance_total.append(relevance)
    return {
        'hit_rate': hit_rate(relevance_total),
        'mrr': mrr(relevance_total),
    }

# --- Решение для Q1 ---

# 4. Индексация документов с помощью minsearch
print("Индексируем документы...")
index = minsearch.Index(
    text_fields=["question", "section", "text"],
    keyword_fields=["course", "id"]
)
index.fit(documents)

# 5. Определение функции поиска с параметрами из Q1
def search_function_q1(q):
    # Параметры, указанные в задании
    boost = {'question': 1.5, 'section': 0.1}
    
    results = index.search(
        query=q['question'],
        filter_dict={'course': q['course']},
        boost_dict=boost,
        num_results=5 # Ограничимся 5 результатами для оценки
    )
    return results

# 6. Оценка и вывод результата
print("Оцениваем качество поиска...")
evaluation_results = evaluate(ground_truth, search_function_q1)
print("\n--- Результат для Вопроса 1 ---")
print(evaluation_results)

# --- Решение для Q2 ---

# 1. Импорт необходимых библиотек
from minsearch import VectorSearch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline

# 2. Создание эмбеддингов ТОЛЬКО для поля "question"
print("\nСоздаем эмбеддинги для вопросов (Q2)...")
texts_q2 = []
for doc in documents:
    # Используем только текст вопроса
    t = doc['question']
    texts_q2.append(t)

# Создаем пайплайн для векторизации, как указано в задании
pipeline_q2 = make_pipeline(
    TfidfVectorizer(min_df=3),
    TruncatedSVD(n_components=128, random_state=1)
)
X_q2 = pipeline_q2.fit_transform(texts_q2)
print(f"Эмбеддинги созданы. Размерность матрицы: {X_q2.shape}")

# 3. Индексация эмбеддингов с помощью VectorSearch
print("Индексируем эмбеддинги...")
vindex_q2 = VectorSearch(keyword_fields={'course'})
vindex_q2.fit(X_q2, documents)

# 4. Определение функции для векторного поиска
def search_function_q2(q):
    # Преобразуем текст вопроса из ground_truth в вектор
    # Важно: используем .transform, а не .fit_transform, чтобы применить уже обученную модель
    question_vector = pipeline_q2.transform([q['question']])[0]
    
    # Выполняем векторный поиск
    results = vindex_q2.search(
        query_vector=question_vector,
        filter_dict={'course': q['course']},
        num_results=5 # Ограничимся 5 результатами
    )
    return results

# 5. Оценка и вывод результата
print("Оцениваем качество векторного поиска (Q2)...")
evaluation_results_q2 = evaluate(ground_truth, search_function_q2)
print("\n--- Результат для Вопроса 2 ---")
print(evaluation_results_q2)

# --- Решение для Q3 ---

# 1. Создание эмбеддингов для "question" + "text"
print("\nСоздаем эмбеддинги для вопросов и ответов (Q3)...")
texts_q3 = []
for doc in documents:
    # Объединяем вопрос и ответ через пробел
    t = doc['question'] + ' ' + doc['text']
    texts_q3.append(t)

# Создаем пайплайн для векторизации (параметры те же)
# Мы можем переиспользовать pipeline_q2, но для ясности создадим новый
pipeline_q3 = make_pipeline(
    TfidfVectorizer(min_df=3),
    TruncatedSVD(n_components=128, random_state=1)
)
X_q3 = pipeline_q3.fit_transform(texts_q3)
print(f"Эмбеддинги созданы. Размерность матрицы: {X_q3.shape}")

# 3. Индексация новых эмбеддингов
print("Индексируем эмбеддинги (Q3)...")
vindex_q3 = VectorSearch(keyword_fields={'course'})
vindex_q3.fit(X_q3, documents)

# 4. Определение функции для поиска (аналогично Q2)
def search_function_q3(q):
    # Важный момент: для поиска мы используем только вопрос из ground_truth.
    # Мы не можем использовать ответ, так как мы его "ищем".
    # Но этот вопрос будет преобразован в вектор с помощью пайплайна,
    # обученного на связках "вопрос-ответ".
    question_vector = pipeline_q3.transform([q['question']])[0]
    
    results = vindex_q3.search(
        query_vector=question_vector,
        filter_dict={'course': q['course']},
        num_results=5
    )
    return results

# 5. Оценка и вывод результата
print("Оцениваем качество векторного поиска (Q3)...")
evaluation_results_q3 = evaluate(ground_truth, search_function_q3)
print("\n--- Результат для Вопроса 3 ---")
print(evaluation_results_q3)


# --- Решение для Q4 (ИСПРАВЛЕННАЯ ВЕРСИЯ) ---

# 1. Импорт библиотек (если они еще не импортированы)
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

# 2. Загрузка модели (если еще не загружена)
print("\nЗагружаем нейросетевую модель (если еще не загружена)...")
model = SentenceTransformer("jinaai/jina-embeddings-v2-small-en", trust_remote_code=True)
print("Модель загружена.")

# 3. Создание эмбеддингов с ПРЕФИКСАМИ
print("Создаем нейросетевые эмбеддинги для документов с префиксами...")
# Добавляем префикс для документов
doc_texts_q4_fixed = ["search_document: " + doc['question'] + ' ' + doc['text'] for doc in documents]
doc_embeddings_q4_fixed = model.encode(doc_texts_q4_fixed, show_progress_bar=True)
print(f"Эмбеддинги для документов созданы. Размерность: {doc_embeddings_q4_fixed.shape}")

# 4. Настройка и заполнение Qdrant
client = QdrantClient(":memory:") 

client.recreate_collection(
    collection_name="faq_documents_fixed", # Используем новое имя коллекции
    vectors_config=models.VectorParams(
        size=doc_embeddings_q4_fixed.shape[1], 
        distance=models.Distance.COSINE
    )
)

print("Загружаем данные в Qdrant...")
client.upload_records(
    collection_name="faq_documents_fixed",
    records=[
        models.Record(
            id=idx,
            vector=vector.tolist(),
            payload=doc
        )
        for idx, (vector, doc) in enumerate(zip(doc_embeddings_q4_fixed, documents))
    ]
)
print("Данные загружены в Qdrant.")


# 5. Определение функции поиска с ПРЕФИКСОМ
def search_function_q4_fixed(q):
    # Добавляем префикс для поискового запроса
    query_text = "search_query: " + q['question']
    query_vector = model.encode(query_text)
    
    search_results = client.search(
        collection_name="faq_documents_fixed",
        query_vector=query_vector,
        query_filter=models.Filter(
            must=[models.FieldCondition(key="course", match=models.MatchValue(value=q['course']))]
        ),
        limit=5
    )
    
    results_list = [hit.payload for hit in search_results]
    return results_list

# 6. Оценка и вывод результата
print("Оцениваем качество поиска с Qdrant и префиксами (Q4 исправленный)...")
evaluation_results_q4_fixed = evaluate(ground_truth, search_function_q4_fixed)
print("\n--- Результат для Вопроса 4 (ИСПРАВЛЕННЫЙ) ---")
print(evaluation_results_q4_fixed)

# --- Решение для Q5 ---

# 1. Импорт необходимых библиотек
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline

# 2. Загрузка данных с результатами RAG
print("\nЗагружаем данные с результатами RAG...")
url_prefix = 'https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/03-evaluation/'
results_url = url_prefix + 'rag_evaluation/data/results-gpt4o-mini.csv'
df_results = pd.read_csv(results_url)
print("Данные загружены.")

# 3. Создание и обучение пайплайна для векторизации
# Как указано в задании, обучаем на всех текстах сразу
all_texts = df_results['answer_llm'].fillna('') + ' ' + \
            df_results['answer_orig'].fillna('') + ' ' + \
            df_results['question'].fillna('')

print("Обучаем пайплайн для векторизации (TF-IDF + SVD)...")
pipeline_q5 = make_pipeline(
    TfidfVectorizer(min_df=3),
    TruncatedSVD(n_components=128, random_state=1)
)
pipeline_q5.fit(all_texts)
print("Пайплайн обучен.")

# 4. Создание эмбеддингов для пар ответов
print("Создаем эмбеддинги для ответов LLM и оригинальных ответов...")
# Используем .transform() для получения векторов
vectors_llm = pipeline_q5.transform(df_results['answer_llm'].fillna(''))
vectors_orig = pipeline_q5.transform(df_results['answer_orig'].fillna(''))

# 5. Вычисление косинусного сходства для каждой пары
def cosine_similarity(u, v):
    # Убедимся, что векторы - это numpy-массивы
    u = np.array(u)
    v = np.array(v)
    
    # Вычисляем норму (длину) каждого вектора
    u_norm = np.linalg.norm(u)
    v_norm = np.linalg.norm(v)
    
    # Избегаем деления на ноль, если вектор нулевой
    if u_norm == 0 or v_norm == 0:
        return 0.0
        
    # Вычисляем косинусное сходство по формуле
    return np.dot(u, v) / (u_norm * v_norm)

print("Вычисляем косинусное сходство...")
cosine_scores = []
for i in range(len(vectors_llm)):
    score = cosine_similarity(vectors_llm[i], vectors_orig[i])
    cosine_scores.append(score)

# 6. Расчет и вывод среднего значения
average_cosine = np.mean(cosine_scores)

print("\n--- Результат для Вопроса 5 ---")
print(f"Среднее косинусное сходство: {average_cosine}")

# --- Решение для Q6 ---

# 1. Импорт необходимых библиотек
from rouge import Rouge
import numpy as np
import pandas as pd # Убедимся, что pandas импортирован

# Если df_results не определен, раскомментируйте следующие строки:
# print("\nЗагружаем данные с результатами RAG (если нужно)...")
# url_prefix = 'https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/03-evaluation/'
# results_url = url_prefix + 'rag_evaluation/data/results-gpt4o-mini.csv'
# df_results = pd.read_csv(results_url)
# df_results = df_results.fillna('') # Заполняем пропуски
# print("Данные загружены.")


# 2. Инициализация объекта для подсчета ROUGE
rouge_scorer = Rouge()

# 3. Вычисление ROUGE-1 F1-score для каждой пары ответов
print("\nВычисляем ROUGE-1 F1-score для всех пар ответов...")
rouge_scores = []

for _, row in df_results.iterrows():
    answer_llm = row['answer_llm']
    answer_orig = row['answer_orig']
    
    # Проверка на пустые строки, чтобы избежать ошибок от библиотеки rouge
    if not answer_llm or not answer_orig:
        # Если один из ответов пустой, сходство равно 0
        rouge_scores.append(0.0)
        continue
    
    # Считаем ROUGE
    # get_scores возвращает список, берем первый элемент [0]
    scores = rouge_scorer.get_scores(answer_llm, answer_orig)[0]
    
    # Извлекаем F1-score для rouge-1
    rouge_1_f1 = scores['rouge-1']['f']
    rouge_scores.append(rouge_1_f1)

# 4. Расчет и вывод среднего значения
average_rouge_f1 = np.mean(rouge_scores)

print("\n--- Результат для Вопроса 6 ---")
print(f"Средний ROUGE-1 F1-score: {average_rouge_f1}")

# Пример для 10-го элемента, как в задании (просто для проверки)
# r = df_results.iloc[10]
# scores_10 = rouge_scorer.get_scores(r.answer_llm, r.answer_orig)[0]
# print(f"\nПример ROUGE для 10-й строки: {scores_10}")
