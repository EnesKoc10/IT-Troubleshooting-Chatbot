FROM python:3.11-slim

WORKDIR /app

# System deps for sentence-transformers/torch and qdrant
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy vectordatabase source and knowledge base
COPY . /app

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    QDRANT_URL=http://qdrant:6333 \
    QDRANT_PORT=6333 \
    QDRANT_COLLECTION=it_support_kb

# Build the knowledge base on startup
CMD ["python", "buildRAG.py", "build", "--pdf-dir", "/app", "--pattern", "*.md", "--qdrant-url", "http://qdrant:6333", "--qdrant-collection", "it_support_kb"]
