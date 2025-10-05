# Stage 1: Base Image with GPU support and Python
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

LABEL maintainer="Your Name <your.email@example.com>"
LABEL description="ETL pipeline for Clinical Recommendations RAG project."

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install system dependencies, with Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Stage 2: Install Python dependencies
WORKDIR /app
COPY requirements.txt .

# Install the production dependencies. This will be much faster and smaller.
RUN pip install --no-cache-dir -r requirements.txt

# Stage 3: Copy the application code
COPY ./etl ./etl

ENTRYPOINT ["python", "-m", "etl.main"]
