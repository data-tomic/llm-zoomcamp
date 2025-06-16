import requests

docs_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main/01-intro/documents.json?raw=1'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()

documents = [] # <- Вот здесь объявляется переменная documents

for course_in_documents_raw in documents_raw: # Используем другое имя для переменной цикла, чтобы избежать путаницы
    course_name = course_in_documents_raw['course']
    for doc in course_in_documents_raw['documents']:
        doc['course'] = course_name
        documents.append(doc)

print(f"Successfully downloaded {len(documents)} documents.")


from elasticsearch import Elasticsearch


es_client = Elasticsearch('http://localhost:9200')

index_name = "course-questions"

index_settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "section": {"type": "text"},
            "question": {"type": "text"},
            "course": {"type": "keyword"}
        }
    }
}

# Проверяем, существует ли индекс, и если нет - создаем
if not es_client.indices.exists(index=index_name):
    es_client.indices.create(index=index_name, body=index_settings)
    print(f"Index '{index_name}' created.")
else:
    print(f"Index '{index_name}' already exists.")

# Индексируем документы
print("Indexing documents...")
for i, doc in enumerate(documents):
    try:
        es_client.index(index=index_name, document=doc, id=i) # Используем 'i' как id документа
    except Exception as e:
        print(f"Error indexing document {i}: {e}")
        # print(doc) # Раскомментируйте для отладки, если будут ошибки с конкретными документами

print(f"Successfully indexed {len(documents)} documents into '{index_name}'.")

# Проверим, что данные действительно есть (опционально)
# response = es_client.count(index=index_name)
# print(f"Number of documents in index '{index_name}': {response['count']}")

print("\n--- Running Q3 Search ---") # Добавим разделитель для ясности вывода
query_text_q3 = "How do execute a command on a Kubernetes pod?"

search_query_q3 = {
    "size": 1,
    "query": {
        "multi_match": {
            "query": query_text_q3,
            "fields": ["question^4", "text"],
            "type": "best_fields"
        }
    }
}

response_q3 = es_client.search(index=index_name, body=search_query_q3)

if response_q3['hits']['hits']: # Проверяем, что есть хотя бы один результат
    top_hit_q3 = response_q3['hits']['hits'][0]
    score_q3 = top_hit_q3['_score']

    print(f"Query: '{query_text_q3}'")
    print(f"Top ranking result score (Q3): {score_q3}")
    # print(f"Top ranking result (Q3): {top_hit_q3['_source']}")
else:
    print(f"No results found for query (Q3): '{query_text_q3}'")


print("\n--- Running Q4 Search ---")
query_text_q4 = "How do copy a file to a Docker container?"
course_filter_q4 = "machine-learning-zoomcamp"

search_query_q4 = {
    "size": 3, # Нам нужно 3 результата
    "query": {
        "bool": {
            "must": {
                "multi_match": {
                    "query": query_text_q4,
                    "fields": ["question^4", "text"], # Используем те же поля и буст, что и в Q3 (или можно question^3 как в примере из описания ДЗ)
                                                    # В описании ДЗ для "Query" в конце есть пример с "question^3".
                                                    # Давайте используем "question^3", "text", "section" как в примере ДЗ.
                                                    # Но в Q3 было "question^4", "text". 
                                                    # Давайте для Q4 будем следовать структуре запроса из описания ДЗ, но поля из задания Q3, если не указано иное.
                                                    # Задание Q4 говорит только "ask a different question", не меняя способ поиска кроме фильтра и кол-ва.
                                                    # Оставим ["question^4", "text"] и "best_fields" для multi_match, так как это было условием для Q3,
                                                    # а Q4 только меняет запрос и добавляет фильтр.
                    "type": "best_fields"
                }
            },
            "filter": {
                "term": {
                    "course": course_filter_q4
                }
            }
        }
    }
}

response_q4 = es_client.search(index=index_name, body=search_query_q4)

print(f"Query: '{query_text_q4}', Filter: course='{course_filter_q4}'")
print(f"Found {len(response_q4['hits']['hits'])} results for Q4.")

if len(response_q4['hits']['hits']) >= 3:
    third_question_q4 = response_q4['hits']['hits'][2]['_source']['question'] # [2] для третьего результата (индексация с 0)
    print(f"The 3rd question returned (Q4): '{third_question_q4}'")
    
    # Распечатаем все 3 вопроса для сверки:
    print("All 3 returned questions for Q4:")
    for i, hit in enumerate(response_q4['hits']['hits']):
        print(f"  {i+1}. {hit['_source']['question']} (Score: {hit['_score']})")
else:
    print("Less than 3 results returned for Q4, cannot determine the 3rd question.")
    if len(response_q4['hits']['hits']) > 0:
        print("Returned questions for Q4:")
        for i, hit in enumerate(response_q4['hits']['hits']):
            print(f"  {i+1}. {hit['_source']['question']} (Score: {hit['_score']})")


print("\n--- Building Prompt for Q5 ---")

context_template = """
Q: {question}
A: {text}
""".strip()

# Формируем строку контекста из результатов Q4
context_entries = []
if 'hits' in response_q4 and 'hits' in response_q4['hits']:
    for hit in response_q4['hits']['hits']: # Берем все 3 документа, полученные в Q4
        source_doc = hit['_source']
        question_text = source_doc['question']
        answer_text = source_doc['text']
        context_entries.append(context_template.format(question=question_text, text=answer_text))

context_str_q5 = "\n\n".join(context_entries)

# Исходный вопрос для Q5 (тот же, что и для поиска в Q4)
question_q5 = "How do copy a file to a Docker container?"

prompt_template = """
You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT:
{context}
""".strip()

final_prompt_q5 = prompt_template.format(question=question_q5, context=context_str_q5)

prompt_length_q5 = len(final_prompt_q5)

print(f"Length of the resulting prompt (Q5): {prompt_length_q5}")
# print("\n--- Resulting Prompt Q5 ---") # Раскомментируйте для просмотра самого промпта
# print(final_prompt_q5)
# print("--- End of Prompt Q5 ---")

# ... (код для Q5, где вы получили final_prompt_q5) ...
# Предполагается, что final_prompt_q5 содержит итоговый промпт из Q5

print("\n--- Calculating Tokens for Q6 ---")

try:
    import tiktoken
except ImportError:
    print("tiktoken library is not installed. Please install it using: pip install tiktoken")
    # Можно здесь выйти или установить значение по умолчанию, если не хотите прерывать скрипт
    # num_tokens_q6 = -1 # или какое-то другое значение, чтобы показать ошибку
    exit() # Лучше прервать, чтобы пользователь установил зависимость

# Загружаем кодировку для модели "gpt-4o"
encoding = tiktoken.encoding_for_model("gpt-4o")

# Кодируем наш промпт (final_prompt_q5 из Q5) в токены
tokens_q6 = encoding.encode(final_prompt_q5)

# Считаем количество токенов
num_tokens_q6 = len(tokens_q6)

print(f"The prompt (from Q5) has {num_tokens_q6} tokens (Q6).")

# Дополнительно: можно посмотреть на некоторые токены и их декодированное значение
# print(f"First 10 tokens: {tokens_q6[:10]}")
# for token_id in tokens_q6[:3]:
#     print(f"Token ID: {token_id}, Decoded: {encoding.decode_single_token_bytes(token_id)}")

# КОД ДЛЯ БОНУСНОГО ВОПРОСА С OLLAMA (убедитесь, что он есть в вашем файле)
print("\n--- Generating Answer with Ollama (Bonus) ---")

# Убедитесь, что переменная final_prompt_q5 доступна из предыдущих шагов
# Если нет, вы можете ее здесь заново определить или передать.
# Мы будем использовать final_prompt_q5, который содержит и контекст и вопрос.

selected_ollama_model = "llama3:8b"
ollama_api_url = "http://localhost:11434/api/generate"

data_payload = {
    "model": selected_ollama_model,
    "prompt": final_prompt_q5,
    "stream": False
}

try:
    import requests
    import json

    print(f"Sending prompt to Ollama model: {selected_ollama_model}...")

    response_ollama = requests.post(ollama_api_url, json=data_payload)
    response_ollama.raise_for_status()

    response_data = response_ollama.json()
    generated_answer = response_data.get("response", "No response content found.")

    print("\n--- Generated Answer from Ollama ---")
    print(generated_answer.strip())
    print("--- End of Ollama Answer ---")

except requests.exceptions.ConnectionError:
    print(f"Error: Could not connect to Ollama API at {ollama_api_url}.")
    print("Please ensure Ollama is running and accessible.")
except requests.exceptions.HTTPError as http_err:
    print(f"HTTP error occurred: {http_err}")
    try:
        error_details = response_ollama.json()
        print(f"Error details: {json.dumps(error_details, indent=2)}")
    except json.JSONDecodeError:
        print(f"Error details (raw response): {response_ollama.text}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")


if 'generated_answer' in locals() and 'encoding' in locals() and 'num_tokens_q6' in locals():
    # Токенизируем ответ Ollama
    tokens_ollama_response = encoding.encode(generated_answer)
    num_tokens_ollama_response = len(tokens_ollama_response)

    print(f"Number of input tokens (Q6): {num_tokens_q6}")
    print(f"Number of output tokens (from Ollama's response, tokenized for gpt-4o): {num_tokens_ollama_response}")

    # Цены OpenAI для gpt-4o
    price_input_per_1k_tokens = 0.005  # $
    price_output_per_1k_tokens = 0.015 # $

    # Расчет стоимости для одного такого запроса (ваш конкретный случай)
    cost_input_specific = (num_tokens_q6 / 1000) * price_input_per_1k_tokens
    cost_output_specific = (num_tokens_ollama_response / 1000) * price_output_per_1k_tokens
    total_cost_specific_request = cost_input_specific + cost_output_specific

    print(f"Cost for your specific input ({num_tokens_q6} tokens): ${cost_input_specific:.6f}")
    print(f"Cost for Ollama's output ({num_tokens_ollama_response} tokens): ${cost_output_specific:.6f}")
    print(f"Total cost for this single specific request (if it were OpenAI gpt-4o): ${total_cost_specific_request:.6f}")

    # Расчет стоимости для 1000 таких же запросов
    total_cost_1000_specific_requests = total_cost_specific_request * 1000
    print(f"Total estimated cost for 1000 such specific requests: ${total_cost_1000_specific_requests:.2f}")

else:
    print("Could not perform specific cost calculation: 'generated_answer', 'encoding', or 'num_tokens_q6' not found.")

print("\n--- Summary of Average Cost Calculation (as per homework prompt) ---")
avg_input_tokens = 150
avg_output_tokens = 250
num_requests = 1000

cost_one_avg_request = ((avg_input_tokens / 1000) * price_input_per_1k_tokens) + \
                       ((avg_output_tokens / 1000) * price_output_per_1k_tokens)
total_cost_1000_avg_requests = cost_one_avg_request * num_requests
print(f"Total cost for 1000 requests (average 150 input, 250 output tokens): ${total_cost_1000_avg_requests:.2f}")
