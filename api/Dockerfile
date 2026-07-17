# Botanical Vision API — containerized production service.
# The explicit upgrade over the course's bare-uvicorn/ngrok lab: a real image running
# gunicorn with uvicorn workers.
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt api/requirements.txt
# CPU-only torch keeps the image sane; the extra index serves the CPU wheels.
RUN pip install --no-cache-dir -r api/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY api/ api/
COPY mlops/ mlops/
COPY models/ models/

# Local registry by default. For a real containerized deploy, point this at a shared
# tracking server (MinIO/S3 artifacts — tier B) or Databricks (tier C); with a
# local-file store the artifact paths must be bind-mounted (see `make docker-run`).
ENV MLFLOW_TRACKING_URI=sqlite:///mlops/mlflow.db

EXPOSE 8000
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "2", "-b", "0.0.0.0:8000", "api.main:app"]
