# LLM Zoomcamp Homework Repository

Welcome to my repository for the [LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp) by DataTalks.Club! This repository contains my solutions and notes for the homework assignments completed throughout the course.

The goal of this zoomcamp is to provide a practical, hands-on introduction to Large Language Models (LLMs), covering topics from fundamental concepts to building and deploying LLM-powered applications.

## About This Repository

This repository is organized by module or homework assignment. Each assignment typically resides in its own subdirectory, containing:
*   Python scripts (`.py`) with the code solutions.
*   Jupyter Notebooks (`.ipynb`) if used for experimentation or presentation.
*   A `README.md` specific to that assignment, detailing the problem, steps taken, and results.
*   Any necessary data files or configuration files.

## Homework Assignments

Below is a list of the homework assignments completed. Click on the links to navigate to the specific homework directory.

| Module / Week | Homework Topic                                  | Directory                                            | Status      |
|---------------|-------------------------------------------------|------------------------------------------------------|-------------|
| 01 - Intro    | Introduction to LLMs, Search & Elasticsearch    | [01-intro](./01-intro/)                              | Completed   |
| 02 - Vector Search | Vector Search with Qdrant & fastembed        | [02-vector-search](./02-vector-search/)              | Completed   |
| 03 - RAG      | Retrieval Augmented Generation (RAG)            | [03-rag](./03-rag/)                                  | To Do       |
| 04 - Fine-tuning | Fine-tuning LLMs                             | [04-fine-tuning](./04-fine-tuning/)                   | To Do       |
| 05 - Deployment | Deploying LLM Applications                     | [05-deployment](./05-deployment/)                     | To Do       |
| 06 - Projects | Capstone Project Work                          | [06-projects](./06-projects/)                         | To Do       |
| ...           | *(More modules/homeworks as they are completed)* | *(Link to directory)*                                | *(Status)*  |


## General Setup and Tools

While each homework might have specific requirements, a common setup often includes:

1.  **Python Environment:**
    *   It's highly recommended to use a virtual environment (e.g., `venv` or `conda`) for each project or for the entire zoomcamp.
    *   Example using `venv`:
        ```bash
        python -m venv .venv
        source .venv/bin/activate  # On Linux/macOS
        # .\.venv\Scripts\Activate.ps1 # On Windows PowerShell
        ```
2.  **Common Libraries:**
    *   `requests`: For making HTTP requests.
    *   `python-dotenv`: For managing environment variables.
    *   `numpy`, `pandas`: For data manipulation.
    *   Specific LLM libraries: `openai`, `ollama`, `langchain`, `llama-index`, `sentence-transformers`, `transformers`, `datasets`, `peft`, `bitsandbytes`, etc.
    *   Search/Vector DB clients: `elasticsearch`, `qdrant-client`, `chromadb`, etc.
    *   Refer to the `requirements.txt` file within each homework directory for specific dependencies.
3.  **Docker:**
    *   Often used for running services like Elasticsearch, vector databases, or even LLM inference containers. Ensure Docker Desktop or Docker Engine is installed and running.
4.  **Ollama:**
    *   For running LLMs locally. Installation instructions can be found on the [Ollama website](https://ollama.com/).
5.  **API Keys:**
    *   Some homeworks might require API keys for services like OpenAI, Cohere, etc. These should be managed securely, for example, using `.env` files and `python-dotenv`. **Never commit API keys directly to your repository.** Add `.env` to your `.gitignore` file.

## Disclaimer

The solutions and code in this repository represent my personal understanding and approach to the homework assignments. They are intended for learning and reference purposes. There might be multiple ways to solve a problem, and these solutions reflect one such way.

## Acknowledgements

A big thank you to the [DataTalks.Club](https://datatalks.club/) team for organizing this comprehensive and practical LLM Zoomcamp.

