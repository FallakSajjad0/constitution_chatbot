FROM python:3.10-slim

WORKDIR /app

# Install ONLY web/runtime dependencies (NO embeddings / torch)
COPY requirements.web.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

ENV PYTHONUNBUFFERED=1

# Railway provides PORT automatically
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]
