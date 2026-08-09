#!/usr/bin/env bash
# Run golang-migrate against local Docker Postgres.
# Usage:
#   ./deploy/scripts/migrate.sh up
#   ./deploy/scripts/migrate.sh down 1
#   ./deploy/scripts/migrate.sh version

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CMD="${1:-up}"
shift || true

if [[ ! -f .env ]]; then
  echo "No deploy/.env — copying from .env.example"
  cp .env.example .env
fi

set -a
# shellcheck disable=SC1091
source .env
set +a

echo "→ Ensuring postgres is up..."
docker compose up -d postgres

echo "→ migrate $CMD $*"
docker compose --profile tools run --rm migrate "$CMD" "$@"
