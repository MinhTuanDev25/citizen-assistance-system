#!/usr/bin/env bash
# Start the whole local stack in Docker: postgres, migrate, api, web, ai-service,
# minio, qdrant, and (when deploy/.env.speech.local exists) the Speech tunnel.
# Usage:
#   ./deploy/scripts/up.sh            # build + start
#   ./deploy/scripts/up.sh --no-build # start existing images

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -f .env ]]; then
  echo "deploy/.env is missing. Copy .env.example to .env first." >&2
  exit 1
fi

BUILD="--build"
if [[ "${1:-}" == "--no-build" ]]; then
  BUILD=""
fi

ENV_ARGS=(--env-file .env)
PROFILES=(--profile app --profile storage --profile ai --profile qdrant)
SPEECH=0
if [[ -f .env.speech.local ]]; then
  ENV_ARGS+=(--env-file .env.speech.local)
  PROFILES+=(--profile speech)
  SPEECH=1
else
  echo "→ No deploy/.env.speech.local — Speech stays disabled (see .env.speech.example)"
fi

echo "→ docker compose up ${BUILD:-(no build)}"
docker compose "${ENV_ARGS[@]}" "${PROFILES[@]}" up -d $BUILD

if [[ "$SPEECH" == 1 ]]; then
  # network_mode service:api binds to the api container that existed at create
  # time; a rebuilt api needs a fresh sidecar.
  echo "→ recreating speech-tunnel inside the api network namespace"
  docker compose "${ENV_ARGS[@]}" "${PROFILES[@]}" up -d --force-recreate --no-deps speech-tunnel
fi

WEB_PORT="$(grep -E '^WEB_PORT=' .env | cut -d= -f2 || true)"
API_PORT="$(grep -E '^API_PORT=' .env | cut -d= -f2 || true)"
echo
echo "Web: http://127.0.0.1:${WEB_PORT:-5173}"
echo "API: http://127.0.0.1:${API_PORT:-8080}"
if [[ "$SPEECH" == 1 ]]; then
  echo "Speech tunnel health: docker inspect -f '{{.State.Health.Status}}' \$(docker compose ps -q speech-tunnel)"
fi
