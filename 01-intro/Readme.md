# Lab: Introduction to Search with Elasticsearch and LLMs

This lab focuses on learning the fundamentals of search using Elasticsearch and integrating it with Large Language Models (LLMs) to answer questions based on retrieved data. We will index a set of FAQs, perform searches with various criteria, construct a prompt for an LLM, and get a response from a locally deployed model using Ollama.

## Table of Contents
1.  [Lab Objectives](#lab-objectives)
2.  [Required Tools and Software](#required-tools-and-software)
3.  [Execution Steps](#execution-steps)
    *   [3.1. Environment Setup](#31-environment-setup)
    *   [3.2. Running Elasticsearch](#32-running-elasticsearch)
    *   [3.3. Running Ollama and Loading a Model](#33-running-ollama-and-loading-a-model)
    *   [3.4. Executing the Script](#34-executing-the-script)
4.  [Answers to Lab Questions](#answers-to-lab-questions)
    *   [Q1. Running Elasticsearch](#q1-running-elasticsearch)
    *   [Q2. Indexing the Data](#q2-indexing-the-data)
    *   [Q3. Searching](#q3-searching)
    *   [Q4. Filtering](#q4-filtering)
    *   [Q5. Building a Prompt](#q5-building-a-prompt)
    *   [Q6. Tokens](#q6-tokens)
    *   [Bonus: Generating the Answer](#bonus-generating-the-answer)
    *   [Bonus: Calculating Costs](#bonus-calculating-costs)
5.  [Structure of `faq_indexing_search.py` Script](#structure-of-faq_indexing_searchpy-script)
6.  [Example `requirements.txt` File](#example-requirementstxt-file)

## Lab Objectives
*   Understand basic Elasticsearch operations: creating an index, mapping, and indexing documents.
*   Execute search queries with various parameters (multi-match, boost, filter).
*   Formulate context for an LLM based on search results.
*   Interact with a locally deployed LLM via Ollama.
*   Estimate token counts and potential costs of cloud-based LLM API requests.

## Required Tools and Software
*   Python 3.8+
*   pip (Python package manager)
*   `venv` (for creating virtual environments)
*   Docker (for running Elasticsearch)
*   `curl` (or a similar tool for HTTP requests, to check Elasticsearch)
*   Ollama (for running LLMs locally)
    *   An installed model (e.g., `llama3:8b` or `phi3:mini`)
*   Text editor or IDE (e.g., VS Code)
*   Terminal/Command Prompt

## Execution Steps

### 3.1. Environment Setup
1.  **Clone the repository (if it already exists) or create a new project directory.**
    ```bash
    # mkdir llm_homework_project
    # cd llm_homework_project
    ```
2.  **Create and activate a Python virtual environment:**
    ```bash
    python -m venv .venv
    # For Linux/macOS
    source .venv/bin/activate
    # For Windows (PowerShell)
    # .\.venv\Scripts\Activate.ps1
    # For Windows (cmd.exe)
    # .\.venv\Scripts\activate.bat
    ```
3.  **Install the required Python libraries:**
    Create a `requirements.txt` file (an example of its content is provided below) and run:
    ```bash
    pip install -r requirements.txt
    ```
    Or install the packages individually:
    ```bash
    pip install requests elasticsearch==8.13.1 tiktoken
    ```
    *(Note: We used `elasticsearch` client version 8.13.1 for compatibility with Elasticsearch server 8.17.6. You can choose another 8.x version if needed.)*

### 3.2. Running Elasticsearch
1.  **Ensure Docker Desktop or Docker Engine is running.**
2.  **Run the Elasticsearch 8.17.6 container with the following command in your terminal:**
    ```bash
    docker run -it \
        --rm \
        --name elasticsearch \
        -p 9200:9200 \
        -p 9300:9300 \
        -e "discovery.type=single-node" \
        -e "xpack.security.enabled=false" \
        docker.elastic.co/elasticsearch/elasticsearch:8.17.6
    ```
3.  **Verify that Elasticsearch is running and accessible:**
    Open a new terminal and execute:
    ```bash
    curl localhost:9200
    ```
    You should see a JSON response with cluster information, including `version.number: "8.17.6"`.

### 3.3. Running Ollama and Loading a Model
This part is necessary for the bonus question on LLM answer generation.
1.  **Install Ollama:**
    Follow the instructions on the [official Ollama website](https://ollama.com/). For Linux/WSL (Ubuntu), the command is usually:
    ```bash
    curl -fsSL https://ollama.com/install.sh | sh
    ```
2.  **Download an LLM model (e.g., `llama3:8b` or `phi3:mini`):**
    ```bash
    ollama pull llama3:8b 
    # or
    # ollama pull phi3:mini
    ```
3.  **Ensure Ollama is running and the model is available:**
    ```bash
    ollama list
    ```
    You should see the downloaded model in the list. The Ollama server usually starts automatically or on the first command.

### 3.4. Executing the Script
1.  Place the `faq_indexing_search.py` file in the root directory of your project.
2.  Ensure your virtual environment is activated, and Elasticsearch and Ollama (if used) are running.
3.  Run the script:
    ```bash
    python faq_indexing_search.py
    ```
    The script will perform all steps: data loading, indexing, searching, prompt building, token counting, answer generation (if enabled), and cost calculation.

## Answers to Lab Questions

### Q1. Running Elasticsearch
**Question:** Run Elastic Search 8.17.6, and get the cluster information. What's the `version.build_hash` value?
**Answer:** `dbcbbbd0bc4924cfeb28929dc05d82d662c527b7`
*(Obtained from the `curl localhost:9200` output)*
```json
{
  "name" : "bfec4cf81ba5", // Your container name might differ
  "cluster_name" : "docker-cluster",
  "cluster_uuid" : "qULYOjUrRbSGwO_Au1SAsw", // Your cluster UUID might differ
  "version" : {
    "number" : "8.17.6",
    "build_flavor" : "default",
    "build_type" : "docker",
    "build_hash" : "dbcbbbd0bc4924cfeb28929dc05d82d662c527b7",
    "build_date" : "2025-04-30T14:07:12.231372970Z",
    "build_snapshot" : false,
    "lucene_version" : "9.12.0",
    "minimum_wire_compatibility_version" : "7.17.0",
    "minimum_index_compatibility_version" : "7.0.0"
  },
  "tagline" : "You Know, for Search"
}
```

### Q2. Indexing the Data
**Question:** Which function do you use for adding your data to elastic?
*   insert
*   **index**
*   put
*   add
**Answer:** `index` (in the Python client, it's `es_client.index(...)`)

### Q3. Searching
**Question:** We will execute a query "How do execute a command on a Kubernetes pod?". Use only `question` and `text` fields and give `question` a boost of 4, and use `"type": "best_fields"`. What's the score for the top ranking result?
*   84.50
*   64.50
*   **44.50**
*   24.50
**Script Result (may vary slightly):** `44.50556` (in the last run)
**Closest Option and Answer:** `44.50`

### Q4. Filtering
**Question:** Now ask a different question: "How do copy a file to a Docker container?". This time we are only interested in questions from `machine-learning-zoomcamp`. Return 3 results. What's the 3rd question returned by the search engine?
*   How do I debug a docker container?
*   **How do I copy files from a different folder into docker container’s working directory?**
*   How do Lambda container images work?
*   How can I annotate a graph?
**Script Result (3rd question):** `How do I copy files from a different folder into docker container’s working directory?`
**Answer:** `How do I copy files from a different folder into docker container’s working directory?`

### Q5. Building a Prompt
**Question:** Take the records returned from Elasticsearch in Q4 and use this template to build the context. (...) What's the length of the resulting prompt?
*   946
*   **1446**
*   1946
*   2446
**Script Result (prompt length):** `1446`
**Answer:** `1446`

### Q6. Tokens
**Question:** Let's calculate the number of tokens in our query (prompt from Q5) using `tiktoken` for `gpt-4o`. How many tokens does our prompt have?
*   120
*   220
*   **320**
*   420
**Script Result (token count):** `320`
**Answer:** `320`

### Bonus: Generating the Answer
**LLM Question (formulated in Q5):**
```
You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: How do copy a file to a Docker container?

CONTEXT:
Q: How do I debug a docker container?
A: Launch the container image in interactive mode and overriding the entrypoint, so that it starts a bash session.
docker run -it --entrypoint bash <image>
If the container is already running, execute a command in the specific container:
docker ps (find the container-id)
docker exec -it <container-id> bash
(Marcos MJD)

Q: How do I copy files from my local machine to docker container?
A: You can copy files from your local machine to docker container using the docker cp command. You can call it in this way:
docker cp /path/to/local/file_or_directory container_id:/path/in/container
(Tr เด่น Trọng Thanh) Thanks! useful

Q: How do I copy files from a different folder into docker container’s working directory?
A: You can copy files from a different folder into the docker container’s working directory using the docker cp command. You need to specify the source path on your local machine and the destination path inside the container. The syntax is as follows:
docker cp /path/to/local/folder/src/* container_id:/app/  
Ensure that container_id is the ID of your running container and /app/ is the working directory (or any other target directory) in the container.
(Navule Pavan Kumar)
```

**Response from Ollama (model: `llama3:8b`, may vary):**
```
Based on the provided context, to copy a file to a Docker container, you can use the `docker cp` command. The basic syntax is:

`docker cp /path/to/local/file_or_directory container_id:/path/in/container`

Where `/path/to/local/file_or_directory` is the location of the file or directory you want to copy from your local machine, and `container_id` is the ID of the running Docker container where you want to copy the file.
```

### Bonus: Calculating Costs
**Premise:** On average, 150 input and 250 output tokens per request. gpt-4o prices: Input: $0.005 / 1K tokens, Output: $0.015 / 1K tokens.
**Question 1:** How much will it cost to run 1000 requests?
**Calculation:**
Cost of one average request = (150/1000 * $0.005) + (250/1000 * $0.015) = $0.00075 + $0.00375 = $0.0045
Total cost for 1000 average requests = $0.0045 * 1000 = **$4.50**

**Question 2:** Recalculate with your values from Q6 (input tokens) and the tokenized Ollama response (output tokens).
**Data from the script:**
*   Input tokens (Q6): `320`
*   Output tokens (Ollama `llama3:8b` response, tokenized for gpt-4o): `94`
**Calculation:**
Cost of your specific request = (320/1000 * $0.005) + (94/1000 * $0.015) = $0.001600 + $0.001410 = $0.003010
Total cost for 1000 such requests = $0.003010 * 1000 = **$3.01**

## Structure of `faq_indexing_search.py` Script
The `faq_indexing_search.py` script performs the following main steps:
1.  **Load Data:** Downloads a JSON file with FAQs and transforms it into a list of documents.
2.  **Initialize Elasticsearch Client:** Connects to the locally running Elasticsearch server.
3.  **Create Index and Mapping (if not exists):** Defines the structure of the `course-questions` index, specifying field types (`text`, `keyword`).
4.  **Index Documents:** Adds all loaded documents to the Elasticsearch index.
5.  **Search for Q3:** Executes a `multi_match` query with a boost for the `question` field.
6.  **Search for Q4:** Executes a `multi_match` query with a filter on the `course` field.
7.  **Build Prompt for Q5:** Forms context from Q4 results and combines it with the main question into a single LLM prompt.
8.  **Count Tokens for Q6:** Uses the `tiktoken` library to determine the number of tokens in the Q5 prompt for the `gpt-4o` model.
9.  **(Bonus) Interact with Ollama:** Sends the Q5 prompt to a locally running LLM (e.g., `llama3:8b` or `phi3:mini`) and prints its response.
10. **(Bonus) Calculate Costs:** Computes the potential cost of OpenAI gpt-4o API requests based on provided average values and on data from Q6 and the Ollama response.

## Example `requirements.txt` File
```
requests
elasticsearch==8.13.1
tiktoken
```
*(The `elasticsearch` version can be another compatible 8.x version)*
