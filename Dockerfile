# Container image for the HTTP service.
#   docker build -t rag-agent-lab .
#   docker run --rm -p 8000:8000 rag-agent-lab
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies are installed before the source is copied, so the slowest layer
# is reused from cache while only the code changes.
COPY pyproject.toml README.md ./
COPY rag_agent ./rag_agent
RUN pip install --no-cache-dir ".[api]"

COPY data ./data

# Run as an unprivileged user: the service never needs root.
RUN useradd --create-home --uid 1000 app && chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "rag_agent.api:app", "--host", "0.0.0.0", "--port", "8000"]
