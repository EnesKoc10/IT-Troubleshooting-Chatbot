FROM python:3.11-slim

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy backend source
COPY . /app

# Create data directory for SQLite
RUN mkdir -p /app/data

# Expose default port
EXPOSE 8000

ENV PYTHONUNBUFFERED=1 \
    UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=8000 \
    QDRANT_URL=http://qdrant:6333 \
    QDRANT_PORT=6333 \
    QDRANT_COLLECTION=it_support_kb

CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "8000"]


