FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

COPY . .

RUN mkdir -p data/raw data/processed data/indices logs \
    && useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 7860

ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=8s --start-period=120s --retries=3 \
    CMD python -c "import os,urllib.request; p=os.environ.get('PORT','7860'); urllib.request.urlopen('http://127.0.0.1:%s/'%p,timeout=5)"

CMD ["python", "-m", "src.ui.app"]
