# Stage 1: Base Image with GPU support and Python
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

LABEL maintainer="Your Name <your.email@example.com>"
LABEL description="ETL pipeline for Clinical Recommendations RAG project."

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
# The following line prevents pip from complaining about running as root.
ENV PIP_ROOT_USER_ACTION=ignore

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

# === THIS IS THE KEY FIX ===
# Install PyTorch and its dependencies first, pointing to the official index URL for CUDA 12.1.
# This downloads a smaller PyTorch binary that links against the system's CUDA libraries
# instead of bundling its own, saving several gigabytes of disk space.
RUN python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121

# Now, install the rest of the production dependencies from our minimal requirements file.
# Pip will see that torch is already installed and will skip it.
RUN python -m pip install --no-cache-dir -r requirements.txt
# ===========================

# Stage 3: Copy the application code
COPY ./etl ./etl

ENTRYPOINT ["python", "-m", "etl.main"]
