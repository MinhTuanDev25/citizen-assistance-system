# Go Backend API (Gin)

Business core: authentication, conversation sessions, Decision Policy Engine,
procedure orchestration, admin knowledge APIs, audit logging.

Call `apps/ai-service` for intent/slot extraction, document processing, RAG, and embeddings.

## Planned layout

```text
apps/api/
├── cmd/api/main.go
├── internal/
│   ├── config/
│   ├── handler/
│   ├── service/
│   ├── repository/
│   └── domain/
├── go.mod
├── Dockerfile
└── README.md
```

Phase 1 target: `POST /api/v1/chat/turns` with Decision Engine + seed procedures.
