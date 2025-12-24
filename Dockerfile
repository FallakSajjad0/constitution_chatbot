FROM python:3.10-slim

WORKDIR /app

# Copy backend requirements
COPY backend/requirements.txt /app/backend/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy full project
COPY . /app

ENV PYTHONUNBUFFERED=1

# Build vector DB (optional but recommended)
RUN python backend/ingest.py || true

# Railway provides PORT automatically
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port $PORT"]
