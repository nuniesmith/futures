# Stage 1: Install dependencies
FROM python:3.11-slim AS deps

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Copy app code and run
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from deps stage
COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy application code
COPY app.py .
COPY cache.py .
COPY api_server.py .

# Create data directory for SQLite journal
RUN mkdir -p /app/data

ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV DB_PATH=/app/data/futures_journal.db

EXPOSE 8501
EXPOSE 8000

CMD ["streamlit", "run", "app.py"]
