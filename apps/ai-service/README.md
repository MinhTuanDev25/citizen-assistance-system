# Python AI Service (FastAPI)

AI workloads: intent/domain detection, structured slot extraction, PDF draft generation,
chunking/embeddings, semantic retrieval (pgvector), grounded response generation.

LLM access goes through a provider abstraction (OpenAI / Gemini / …).

## Planned layout

```text
apps/ai-service/
├── app/
│   ├── main.py
│   ├── api/
│   ├── services/
│   └── providers/
├── requirements.txt
├── Dockerfile
└── README.md
```

Phase 1 target: mock (then LLM) slot extraction API consumed by Go backend.
