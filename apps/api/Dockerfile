# Go API — multi-stage image (go-live)
# Build from repo root:
#   docker build -f apps/api/Dockerfile -t cas-api:latest .
#
# Run (secrets via env at deploy):
#   docker run --rm -p 8080:8080 \
#     -e APP_ENV=prod \
#     -e DATABASE_URL='postgres://...' \
#     cas-api:latest

FROM golang:1.25-alpine AS builder
WORKDIR /src
RUN apk add --no-cache git ca-certificates
COPY apps/api/go.mod apps/api/go.sum ./
RUN go mod download
COPY apps/api/ ./
RUN CGO_ENABLED=0 GOOS=linux go build -o /out/api ./cmd/api

FROM alpine:3.20
WORKDIR /app
RUN apk add --no-cache ca-certificates tzdata
COPY --from=builder /out/api /app/api
COPY --from=builder /src/configs /app/configs

# APP_ENV selects configs/${APP_ENV}/config.yaml. Secrets via env at deploy.
EXPOSE 8080
CMD ["./api"]
