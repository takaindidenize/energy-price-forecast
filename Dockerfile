# Forecast API image. The warehouse is read from $PF_DATA_DIR/warehouse.duckdb; mount it at /data.
#   docker build -t energy-price-forecast .
#   docker run -p 8000:8000 -v "$PWD/data:/data" energy-price-forecast
# The same image runs the daily jobs by overriding the command, e.g.
#   docker run -v "$PWD/data:/data" -e EPIAS_USERNAME -e EPIAS_PASSWORD energy-price-forecast \
#       python -m price_forecast.pipeline
FROM python:3.12-slim

# LightGBM needs the OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PF_DATA_DIR=/data

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

RUN useradd --create-home --uid 1000 app && mkdir -p /data && chown app:app /data
USER app
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

CMD ["uvicorn", "price_forecast.api:app", "--host", "0.0.0.0", "--port", "8000"]
