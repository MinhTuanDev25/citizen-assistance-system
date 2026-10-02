# P2 — LLM Extract: blocker còn lại

**Ngày:** 2026-09-30

Không deploy, không commit, không push. Không gọi OpenAI hay Gemini. PDF, procedure draft, embedding, RAG và speech vẫn ngoài phạm vi.

Kiến trúc turn không đổi: transaction ngắn, gọi AI khi không giữ row lock, rồi transaction ghi. Claim `PENDING` và idempotency migration `000008` giữ nguyên.

## Đã sửa

1. **Rate limit `POST /v1/extract`.** Sau khi kiểm tra service token, trước khi gọi provider. `ExtractLimiter` giới hạn số request trong 60 giây (`EXTRACT_RATE_PER_MINUTE`, mặc định 60) và số provider call đang chạy (`EXTRACT_MAX_INFLIGHT`, mặc định 4). Vượt giới hạn trả 429 với `Retry-After` (`EXTRACT_RETRY_AFTER_SECONDS`, mặc định 1). Request 401 không tiêu quota. Go nhận 429 một lần, không retry, fallback keyword với `extract_fallback_reason=rate_limited`. Body 429 bị bỏ, không log token hay nội dung tin nhắn.

2. **Số cấu hình fail-closed.** Go từ chối `NaN`, `Inf`, `+Inf`, `-Inf` cho `AI_INTENT_SELECT_MIN`, `AI_INTENT_CONFIRM_MIN`, `AI_SLOT_CONFIDENCE_MIN` trước khi so khoảng. Lỗi parse không echo giá trị. Python `LLM_REQUEST_TIMEOUT_S` phải finite; `nan`/`inf`/`-inf`/`+inf` bị từ chối và không xuất hiện trong message lỗi.

3. **Startup.** `LLM_MODEL` dùng cùng `MODEL_RE` và độ dài tối đa 80 với `model` trong response. Rỗng thì lấy default của provider (`mock-extract-v1`, `gpt-4o-mini`, `gemini-1.5-flash`). `AI_SERVICE_TOKEN` phải dài 16–256 khi process Python khởi động. Go từ chối token khác rỗng mà ngắn hơn 16, kể cả khi AI đang tắt. Lỗi không in token hay model bị từ chối.

4. **Một nguồn trần payload.** Đã xóa `EXTRACT_MAX_CANDIDATES`, `EXTRACT_MAX_SLOTS_PER_CANDIDATE`, `EXTRACT_MAX_MESSAGE_CODEPOINTS`, `EXTRACT_MAX_STRING_VALUE_LEN` khỏi `Settings`. Trần nằm ở hằng số Pydantic trong `app/models/extract.py`.

5. **Compose keyword-only.** `ai-service` chỉ thuộc profile `ai`. `api.depends_on.ai-service.required` là `false`. `--profile app` không start `ai-service` và không chờ nó healthy. Bật AI: `AI_EXTRACT_ENABLED=true docker compose --profile app --profile ai up`. `ai-service` vẫn không có host port.

6. **README.** Lệnh chạy local có `AI_SERVICE_TOKEN` ở cả hai process. Mô tả profile, rate limit, kiểm tra model, và cách bật/tắt AI.

## Giữ nguyên

Provider và model do server gán. Go `DisallowUnknownFields` và từ chối JSON thừa. Body tối đa 256 KiB. Cùng `(session_id, request_id, payload_hash)` chỉ một extractor call khi cạnh tranh bình thường. Lease `PENDING` hết hạn thì request sau lấy lại claim. Gọi AI không giữ DB lock. `AI_EXTRACT_ENABLED` mặc định `false`.

## Kiểm tra đã chạy

Chi tiết trong `phase_p2_test_output.txt`.

| Bước | Kết quả |
|---|---|
| `gofmt` trên file Go vừa sửa | không còn diff format |
| `go build ./...` | OK |
| `go vet` và `go vet -tags=integration` | OK |
| `go test ./...` | OK |
| `CAS_INTEGRATION=1 go test -tags=integration` | OK |
| `go test -race -tags=integration` | OK |
| migrate scratch `up` → `down 8` → `up` | version 8, không dirty |
| pytest | 55 passed |
| `compileall` | OK |
| vitest, vite build, bundle check, `npm audit` | 8 tests, 0 vulnerabilities |
| Compose `--profile app` và `--profile app --profile ai` | keyword-only không có `ai-service`; profile `ai` không publish port |

## Không làm

Không build/push image, không deploy, không `git commit`, không `git push`.
