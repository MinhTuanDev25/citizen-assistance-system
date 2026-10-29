# syntax=docker/dockerfile:1.7
# All-in-one image for one RunPod GPU pod (Phase 8.1): nginx, Go API, index-reaper,
# Python AI service, Speech C0, PostgreSQL 16 + pgvector, Qdrant, MinIO, under
# tini -> supervisord. Data lives on the volume at /workspace/cas.
#
# Build from the repo root after staging C0 (see deploy/runpod/README.md):
#   deploy/runpod/stage_c0.sh
#   docker buildx build --platform linux/amd64 -f deploy/runpod/Dockerfile \
#     --build-arg CAS_REVISION=$(git rev-parse HEAD) -t cas-aio:<sha> .
# Context filter: deploy/runpod/Dockerfile.dockerignore (allowlist).

ARG PYTHON_IMAGE=python:3.11-slim-bookworm

# ---- web: static build on the build machine's arch (esbuild crashes under QEMU) ----
FROM --platform=$BUILDPLATFORM node:22-alpine AS web
WORKDIR /src
COPY apps/web/package.json apps/web/package-lock.json ./
RUN npm ci
COPY apps/web/ ./
ENV VITE_ADMIN_INGESTION=true VITE_ADMIN_INDEXING=true
RUN npm run build && node scripts/check-bundle.mjs --on

# ---- go: cross-compiled for the target arch ----
FROM --platform=$BUILDPLATFORM golang:1.25-alpine AS go
ARG TARGETOS TARGETARCH
WORKDIR /src
COPY apps/api/go.mod apps/api/go.sum ./
RUN go mod download
COPY apps/api/ ./
RUN set -e; for cmd in api index-reaper bootstrap-admin; do \
      CGO_ENABLED=0 GOOS=$TARGETOS GOARCH=$TARGETARCH go build -trimpath -o /out/$cmd ./cmd/$cmd; \
    done

# ---- third-party binaries, pinned images (pulled for the target arch) ----
FROM migrate/migrate:v4.18.1 AS migrate
FROM bitnamilegacy/minio:2024.12.18 AS minio
FROM qdrant/qdrant:v1.13.4 AS qdrant

# ---- runtime ----
FROM ${PYTHON_IMAGE} AS runtime
ARG CAS_REVISION=unknown
LABEL org.opencontainers.image.title="citizen-assistance-aio" \
      org.opencontainers.image.revision="${CAS_REVISION}"
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
# Slow or flaky networks made two builds fail on pip downloads (hash mismatch on a
# truncated file, read timeout): longer timeout and retries for every pip install.
ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1 \
    PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10

# Fixed UIDs so ownership on the volume is stable across image versions and pods.
RUN groupadd -g 10001 cas && useradd -u 10001 -g cas -M -d /nonexistent -s /usr/sbin/nologin cas \
 && groupadd -g 10010 postgres && useradd -u 10010 -g postgres -m -d /var/lib/postgresql -s /bin/bash postgres \
 && groupadd -g 10011 minio && useradd -u 10011 -g minio -M -d /nonexistent -s /usr/sbin/nologin minio \
 && groupadd -g 10012 qdrant && useradd -u 10012 -g qdrant -M -d /nonexistent -s /usr/sbin/nologin qdrant \
 && groupadd -g 10013 speech && useradd -u 10013 -g speech -M -d /nonexistent -s /usr/sbin/nologin speech \
 && groupadd -g 10014 web && useradd -u 10014 -g web -M -d /nonexistent -s /usr/sbin/nologin web

# PostgreSQL 16 + pgvector from PGDG (same major as the dev stack). No auto cluster.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl gnupg \
 && curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc | gpg --dearmor -o /usr/share/keyrings/pgdg.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/pgdg.gpg] http://apt.postgresql.org/pub/repos/apt bookworm-pgdg main" \
      > /etc/apt/sources.list.d/pgdg.list \
 && mkdir -p /etc/postgresql-common && echo "create_main_cluster = false" > /etc/postgresql-common/createcluster.conf \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      postgresql-16 postgresql-client-16 postgresql-16-pgvector \
      nginx supervisor tini gosu openssl procps iproute2 libunwind8 \
      tesseract-ocr tesseract-ocr-vie libgomp1 libglib2.0-0 libgl1 libsndfile1 \
 && apt-get purge -y gnupg && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/* /etc/nginx/sites-enabled/default \
 && dpkg-query -W -f='${Package}=${Version}\n' postgresql-16 postgresql-16-pgvector nginx supervisor tini gosu tesseract-ocr \
      > /opt/apt-versions.txt

# Python environments: AI service and Speech C0 are separate venvs (different pins).
COPY apps/ai-service/requirements-prod.txt /opt/cas/req/ai.txt
RUN python -m venv /opt/cas/venv-ai && /opt/cas/venv-ai/bin/pip install --no-cache-dir -r /opt/cas/req/ai.txt
COPY deploy/runpod/build/c0/requirements-c0-inference.txt deploy/runpod/build/c0/requirements-c0-speech-api.txt /opt/cas/req/
RUN python -m venv /opt/cas/venv-c0 \
 && /opt/cas/venv-c0/bin/pip install --no-cache-dir -r /opt/cas/req/requirements-c0-speech-api.txt \
 && /opt/cas/venv-c0/bin/pip freeze > /opt/cas/req/c0-freeze.txt \
 && /opt/cas/venv-ai/bin/pip freeze > /opt/cas/req/ai-freeze.txt

COPY --from=migrate /usr/local/bin/migrate /usr/local/bin/migrate
COPY --from=minio /opt/bitnami/minio/bin/minio /opt/cas/bin/minio
COPY --from=minio /opt/bitnami/minio-client/bin/mc /opt/cas/bin/mc
COPY --from=qdrant /qdrant/qdrant /opt/cas/qdrant/qdrant
COPY --from=qdrant /qdrant/config /opt/cas/qdrant/config
COPY --from=go /out/ /opt/cas/api/
COPY apps/api/configs /opt/cas/api/configs
COPY apps/ai-service/app /opt/cas/ai/app
COPY deploy/migrations /opt/cas/migrations
COPY --from=web /src/dist /opt/cas/web
COPY deploy/runpod/build/c0 /opt/cas/c0
COPY deploy/runpod/rootfs/ /

# The image revision must be the commit C0 was staged from (stage_c0.sh output).
RUN set -e; \
    install -d -o cas -g cas -m 0700 /home/cas; usermod -d /home/cas cas; \
    install -d -o speech -g speech -m 0700 /home/speech; usermod -d /home/speech speech; \
    chmod 0755 /opt/cas/bin/*.sh /opt/cas/bin/cas-run /opt/cas/bin/cas-pg-recover /opt/cas/bin/svc-run; \
    chown -R root:root /opt/cas; \
    chmod -R a+rX,go-w /opt/cas; \
    if [ "${CAS_REVISION}" != unknown ] && [ "$(cat /opt/cas/c0/SOURCE_REVISION)" != "${CAS_REVISION}" ]; then \
      echo "CAS_REVISION ${CAS_REVISION} != staged C0 revision $(cat /opt/cas/c0/SOURCE_REVISION)" >&2; exit 1; \
    fi; \
    echo "${CAS_REVISION}" > /opt/cas/REVISION

ENV PATH=/opt/cas/bin:$PATH
EXPOSE 8088
STOPSIGNAL SIGTERM
HEALTHCHECK --interval=30s --timeout=15s --start-period=600s --retries=3 \
  CMD /opt/cas/bin/healthcheck.sh --quiet
ENTRYPOINT ["/usr/bin/tini", "--", "/opt/cas/bin/entrypoint.sh"]
