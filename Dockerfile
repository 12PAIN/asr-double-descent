FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --break-system-packages --no-cache-dir -r requirements.txt

COPY *.py ./

ENV HF_HOME=/cache/huggingface
ENV PYTHONUNBUFFERED=1
