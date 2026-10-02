# ADR-002: Procedure definition JSON-driven

## Status

Accepted (Phase 0) — publish path updated to proposal §4.2.

## Context

V1 công dân là **một domain hộ tịch**. Vẫn không hardcode từng thủ tục trong Go: xã đổi giấy tờ thì đổi JSON + PDF, không deploy lại engine.

## Decision

Mỗi thủ tục lưu `procedure_versions.definition` JSON: identity, intent examples, slots, required/conditional, questions, guidance, citations, version.

Cán bộ soạn/giữ JSON (seed hoặc sửa). PDF upload → chunk/embed → **activate**. Không PDF→draft LLM, không workspace duyệt nhiều bước.

## Consequences

- Thêm thủ tục hộ tịch = JSON + nguồn PDF, không đổi Decision Engine
- Schema validation trước activate
- `manual_seed` được trước khi có PDF xã; không coi là nguồn pháp lý
