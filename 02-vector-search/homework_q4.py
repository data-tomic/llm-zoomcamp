from fastembed import TextEmbedding
import numpy as np

# --- Код из Q1 для получения query_embedding и embedding_model ---
query = 'I just discovered the course. Can I join now?'
model_name = 'jinaai/jina-embeddings-v2-small-en'
embedding_model = TextEmbedding(model_name=model_name, cache_dir="local_cache")

query_embedding_list = list(embedding_model.embed([query]))
query_embedding = None
if query_embedding_list:
    query_embedding = query_embedding_list[0]
else:
    print("Failed to generate query embedding for Q1.")
    exit()
# --- Конец кода из Q1 ---

# --- Документы из Q3 ---
documents_q3_or_q4 = [
    {'text': "Yes, even if you don't register, you're still eligible to submit the homeworks.\nBe aware, however, that there will be deadlines for turning in the final projects. So don't leave everything for the last minute.",
     'section': 'General course-related questions',
     'question': 'Course - Can I still join the course after the start date?',
     'course': 'data-engineering-zoomcamp'},
    {'text': 'Yes, we will keep all the materials after the course finishes, so you can follow the course at your own pace after it finishes.\nYou can also continue looking at the homeworks and continue preparing for the next cohort. I guess you can also start working on your final capstone project.',
     'section': 'General course-related questions',
     'question': 'Course - Can I follow the course after it finishes?',
     'course': 'data-engineering-zoomcamp'},
    {'text': "The purpose of this document is to capture frequently asked technical questions\nThe exact day and hour of the course will be 15th Jan 2024 at 17h00. The course will start with the first  “Office Hours'' live.1\nSubscribe to course public Google Calendar (it works from Desktop only).\nRegister before the course starts using this link.\nJoin the course Telegram channel with announcements.\nDon’t forget to register in DataTalks.Club's Slack and join the channel.",
     'section': 'General course-related questions',
     'question': 'Course - When will the course start?',
     'course': 'data-engineering-zoomcamp'},
    {'text': 'You can start by installing and setting up all the dependencies and requirements:\nGoogle cloud account\nGoogle Cloud SDK\nPython 3 (installed with Anaconda)\nTerraform\nGit\nLook over the prerequisites and syllabus to see if you are comfortable with these subjects.',
     'section': 'General course-related questions',
     'question': 'Course - What can I do before the course starts?',
     'course': 'data-engineering-zoomcamp'},
    {'text': 'Star the repo! Share it with friends if you find it useful ❣️\nCreate a PR if you see you can improve the text or the structure of the repository.',
     'section': 'General course-related questions',
     'question': 'How can we contribute to the course?',
     'course': 'data-engineering-zoomcamp'}
]
# --- Конец документов ---

# --- Код для Q4 ---
# Создаем полные тексты (вопрос + текст)
full_texts_q4 = [doc['question'] + ' ' + doc['text'] for doc in documents_q3_or_q4]

# Получаем эмбеддинги для полных текстов
full_text_embeddings_q4_generator = embedding_model.embed(full_texts_q4)
full_text_embeddings_q4 = list(full_text_embeddings_q4_generator)

if not all(isinstance(emb, np.ndarray) for emb in full_text_embeddings_q4):
    print("Failed to generate some full_text embeddings for Q4.")
    exit()

# Рассчитываем косинусное сходство
similarities_q4 = [np.dot(query_embedding, doc_emb) for doc_emb in full_text_embeddings_q4]

# Находим индекс документа с наибольшим сходством
highest_similarity_index_q4 = np.argmax(similarities_q4)

print(f"Similarities for Q4 (full_text): {similarities_q4}")
print(f"Index of document with highest similarity for Q4 (full_text): {highest_similarity_index_q4}")

# Сравнение с Q3 (значение из предыдущего вывода)
# highest_similarity_index_q3 = 1 # Вы получили это значение ранее
# if highest_similarity_index_q4 != highest_similarity_index_q3:
#     print(f"The highest scoring document changed from Q3 (index {highest_similarity_index_q3}) to Q4 (index {highest_similarity_index_q4}).")
#     print("This could be because concatenating 'question' and 'text' provides a more comprehensive semantic context for the query, leading to a different document being a better overall match.")
# else:
#     print(f"The highest scoring document (index {highest_similarity_index_q4}) is the same as in Q3.")