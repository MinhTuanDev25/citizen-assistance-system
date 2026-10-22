# AI service. Python 3.11 matches the pinned PaddlePaddle 3.0.0 and ONNX Runtime wheels.
# Build from repo root:
#   docker build -f apps/ai-service/Dockerfile -t cas-ai:latest .
# Model weights are bind-mounted at runtime. This image does not download them.

FROM python:3.11-slim-bookworm
WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*
COPY apps/ai-service/requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt
COPY apps/ai-service/app ./app
EXPOSE 8001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
