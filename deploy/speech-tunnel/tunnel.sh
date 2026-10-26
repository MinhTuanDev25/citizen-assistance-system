#!/bin/sh
# Keep an SSH local forward 127.0.0.1:${SPEECH_TUNNEL_LOCAL_PORT} → RunPod
# 127.0.0.1:${RUNPOD_SPEECH_PORT}, reconnecting on failure.
# RunPod must expose SSH over TCP (root@<ip> -p <port>). The ssh.runpod.io
# proxy does not support port forwarding.

set -eu

: "${RUNPOD_SSH_HOST:?RUNPOD_SSH_HOST is required}"
: "${RUNPOD_SSH_PORT:?RUNPOD_SSH_PORT is required}"
RUNPOD_SSH_USER="${RUNPOD_SSH_USER:-root}"
RUNPOD_SPEECH_PORT="${RUNPOD_SPEECH_PORT:-8090}"
SPEECH_TUNNEL_LOCAL_PORT="${SPEECH_TUNNEL_LOCAL_PORT:-18090}"
KEY_SRC=/run/secrets/runpod_ssh_key
STATE=/state

if [ ! -s "$KEY_SRC" ]; then
  echo "speech-tunnel: SSH key missing or empty at $KEY_SRC (set RUNPOD_SSH_KEY_PATH)" >&2
  exit 1
fi

# Bind-mounted keys keep host permissions; ssh refuses group/other-readable keys.
mkdir -p "$STATE"
cp "$KEY_SRC" /tmp/id_runpod
chmod 0600 /tmp/id_runpod

echo "speech-tunnel: 127.0.0.1:${SPEECH_TUNNEL_LOCAL_PORT} -> ${RUNPOD_SSH_USER}@${RUNPOD_SSH_HOST}:${RUNPOD_SSH_PORT} -> 127.0.0.1:${RUNPOD_SPEECH_PORT}"

while true; do
  ssh -N \
    -i /tmp/id_runpod \
    -p "$RUNPOD_SSH_PORT" \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -o StrictHostKeyChecking=accept-new \
    -o UserKnownHostsFile="$STATE/known_hosts" \
    -o BatchMode=yes \
    -L "127.0.0.1:${SPEECH_TUNNEL_LOCAL_PORT}:127.0.0.1:${RUNPOD_SPEECH_PORT}" \
    "${RUNPOD_SSH_USER}@${RUNPOD_SSH_HOST}" || true
  echo "speech-tunnel: ssh exited; reconnecting in 5s" >&2
  sleep 5
done
