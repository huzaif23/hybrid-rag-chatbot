# syntax=docker/dockerfile:1
# Single service: FastAPI + Vite production build (Railway / Docker).

FROM node:20-alpine AS frontend-build
WORKDIR /fe
COPY @frontend/package.json @frontend/package-lock.json ./
RUN npm ci
COPY @frontend/ ./
RUN npm run build

FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/@backend

ENV PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/tmp/huggingface

COPY @backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY @backend/ ./
COPY --from=frontend-build /fe/dist ./static

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
