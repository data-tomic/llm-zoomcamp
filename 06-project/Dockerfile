# Stage 1: Base Image with GPU support and Python
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

LABEL maintainer="Your Name <your.email@example.com>"
LABEL description="ETL pipeline for Clinical Recommendations RAG project."

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3-venv git libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Stage 2: Install Python dependencies
WORKDIR /app
COPY requirements.txt .

RUN pip install --no-cache-dir paddlepaddle-gpu==2.6.1 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
RUN pip install --no-cache-dir -r requirements.txt

# Stage 3: Copy the application code
COPY ./etl ./etl

ENTRYPOINT ["python", "-m", "etl.main"]
