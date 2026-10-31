#!/usr/bin/env bash
# Runs probe_tests.sh in throwaway containers (Docker): on an ext4 volume and on bindfs
# mimicking the pod's network volume (needs /dev/fuse and SYS_ADMIN for the bindfs mount
# only; the probe itself does not need them). Exit 1 if any test fails.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PROBE="$(cd "$HERE/.." && pwd)"
IMG=python:3.10-bookworm
docker run --rm -v /workspace -v "$PROBE":/probe:ro "$IMG" bash /probe/tests/probe_tests.sh ext4
docker run --rm --cap-add SYS_ADMIN --device /dev/fuse --security-opt apparmor:unconfined -v "$PROBE":/probe:ro \
  "$IMG" bash /probe/tests/probe_tests.sh bindfs
