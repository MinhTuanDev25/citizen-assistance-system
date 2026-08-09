# Architecture V2 — Summary

## 1. Hai pipeline tách biệt

```text
Admin Pipeline                          Citizen Runtime
-------------                           ---------------
Upload PDF/text                         Chat message
   ↓                                       ↓
Extract draft JSON                      Auth + Conversation Manager
   ↓                                       ↓
Human Review Workspace                  Decision Policy Engine
   ↓                                       ↓
Publish + Version + Embeddings          Procedure Orchestrator
   ↓                                    (load JSON, missing slots)
PostgreSQL + Vector DB                     ↓
                                        Knowledge Service + Citation
                                           ↓
                                        Response to citizen
```

## 2. Core principle

Logic nghiệp vụ **không hardcode trong code**.  
Mỗi thủ tục = 1 `procedure_definition` JSON (slots, questions, guidance, citations).  
Runtime chỉ:

1. Detect procedure
2. So slot
3. Quyết định action
4. Hỏi thêm hoặc trả guidance + nguồn

## 3. Layers (Citizen)

| Layer | Responsibility |
|-------|----------------|
| Citizen Portal | Chat UI, history |
| Auth | JWT, user profile |
| Conversation Manager | Intent/domain, extract slots, session context |
| **Decision Policy Engine** | `ASK_MISSING_SLOTS` / `DIRECT_ANSWER` / `PROVIDE_FINAL_GUIDANCE` |
| Procedure Orchestrator | Load active JSON, validate slots, update slot_state |
| Knowledge Service | RAG + natural response + citation |
| Stores | PostgreSQL (session, JSON, versions) + Vector DB |

## 4. Layers (Admin)

| Layer | Responsibility |
|-------|----------------|
| Admin Portal | Upload, review, publish |
| Auth | Admin role |
| Knowledge Processing | PDF → draft JSON (LLM assist) |
| Review Workspace | Validate schema, edit, approve |
| Knowledge Publisher | Versioning, embeddings, activate, rollback |

## 5. Cross-cutting (bắt buộc V1 tối thiểu)

- `xa_id` filter (single xã)
- Citation + audit (procedure_version used)
- Schema validation trước publish
- Timeout/retry LLM (basic)
- Observability: route decision, slot completion

## 6. What is intentionally deferred

- Multi-xã / multi-tenant
- Voice UI
- Online submission / payment
- Full eval harness tự động (Phase 5+)
- 2-step approval phức tạp (V1: 1 admin approve đủ)
