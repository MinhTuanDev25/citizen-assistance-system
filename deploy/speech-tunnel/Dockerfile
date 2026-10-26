# Speech tunnel sidecar — SSH local forward to the RunPod C0 Speech API.
# Runs with network_mode "service:api", so the Go API sees the tunnel on its
# own loopback (http://127.0.0.1:18090). Go config only accepts loopback
# SPEECH_SERVICE_URL; this keeps that rule without code changes.
# Build from deploy/:
#   docker build -t cas-speech-tunnel:local speech-tunnel

FROM alpine:3.20
RUN apk add --no-cache openssh-client
COPY tunnel.sh /usr/local/bin/tunnel.sh
RUN chmod 0755 /usr/local/bin/tunnel.sh
ENTRYPOINT ["/usr/local/bin/tunnel.sh"]
