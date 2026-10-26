# RunPod runbook — Bahnar Speech (C0) và hướng go-live

Một runbook cho mọi thứ chạy trên RunPod. Gộp từ các tài liệu Step 7.3–7.5
(`RUNPOD_DEPLOYMENT`, `RUNPOD_SPEECH_API`, `RUNPOD_GO_SPEECH`) và kế hoạch
go-live một pod.

Trạng thái từng phần được ghi rõ:

| Phần | Trạng thái |
|---|---|
| §1–§4 Speech API C0 trên pod | Đang dùng. Operator đã chạy thật (Step 7.4–7.6) |
| §5 Docker local → RunPod qua `speech-tunnel` | Đang dùng. Operator xác nhận dịch thật 2026-10-26 |
| §6 Go và Speech cùng namespace | Có trong code (`SPEECH_SERVICE_URL=http://127.0.0.1:8090`), chưa chạy trên pod |
| §7 Go-live toàn bộ hệ thống trên một pod | **KẾ HOẠCH, chưa triển khai, chưa kiểm chứng** |

Không train, không ghi đè checkpoint, không commit token, SSH key hay file
runtime YAML chứa đường dẫn máy.

## 1. Layout trên pod

| Vai trò | Đường dẫn / biến |
|---|---|
| CAS source (Git checkout) | `CAS_ROOT=/workspace/citizen-assistance-system` |
| Thesis source | `THESIS_SOURCE_ROOT=$CAS_ROOT/bahnar-s2tt-thesis` |
| Artifact root (checkpoint đã train) | `ARTIFACT_ROOT=/workspace/bahnar-s2tt-thesis` |
| C0 venv | `C0_INFERENCE_VENV=/workspace/.venvs/c0-inference` |
| Runtime YAML của operator | `$THESIS_SOURCE_ROOT/configs/c0_inference.runpod.yaml` (không commit) |

Checkout `/workspace/citizen-assistance-system` có thay đổi RQ2 chưa commit
(notebook 14, `artifacts/rq2`). **Phải giữ.** Không tạo worktree mới (Step 7.4
thay đề xuất worktree của Step 7.3). `scripts/runpod/run_speech_api.sh` tự suy
`CAS_ROOT` từ vị trí của nó (`../..`).

Lệnh cấm trên pod: `git reset --hard`, `git clean`, `git stash` tự động,
ghi đè checkout, copy checkpoint lớn, tải tokenizer Hub hay load ASR/MT khi
chỉ inspection.

### Checkpoint đã chọn (C0 cascaded)

- ASR: `$ARTIFACT_ROOT/bahnar_s2tt/full_state/contract_b5172ee0cc391e4f/checkpoints/full_train/full_xlsr300m_v1/ckpts/checkpoint-12500`
- MT: `$ARTIFACT_ROOT/bahnar_s2tt/mt_full_state/contract_c3d7b95698738634/checkpoints/full_train/mt_bartpho_syllable_v1/ckpts/checkpoint-17500`
- MT contract: `$ARTIFACT_ROOT/bahnar_s2tt/mt_full_state/contract_c3d7b95698738634/mt_training_contract.json`
- Serving tokenizer: `vinai/bartpho-syllable` revision `36eee8b4d648dd99da56462edcda3c5c97f7f3de`
- RQ2 artifact root (không đụng): `$ARTIFACT_ROOT/bahnar_s2tt/direct_full_state`

### Runtime YAML (một lần)

Mẫu duy nhất là `bahnar-s2tt-thesis/configs/c0_inference.example.yaml`:

```bash
SRC=/workspace/citizen-assistance-system/bahnar-s2tt-thesis
cp "$SRC/configs/c0_inference.example.yaml" "$SRC/configs/c0_inference.runpod.yaml"
```

Điền đường dẫn ARTIFACT_ROOT ở trên vào `asr_checkpoint`, `mt_checkpoint`,
`mt_training_contract`, và `device: cuda`.

### Inspection chỉ đọc (không load model)

```bash
export PHASE7_ARTIFACT_ROOT=/workspace/bahnar-s2tt-thesis
bash /workspace/citizen-assistance-system/scripts/runpod/inspect_runpod_phase7.sh
```

## 2. Khởi động Speech API (một worker)

Cài HTTP extras một lần, cùng venv, không đổi pin ML:

```bash
source /workspace/.venvs/c0-inference/bin/activate
python -m pip install -r /workspace/citizen-assistance-system/bahnar-s2tt-thesis/requirements-c0-speech-api.txt
```

Chạy:

```bash
export CAS_ROOT=/workspace/citizen-assistance-system
export C0_INFERENCE_CONFIG="${CAS_ROOT}/bahnar-s2tt-thesis/configs/c0_inference.runpod.yaml"
export SPEECH_SERVICE_TOKEN='<token, >= 16 ký tự>'
export SPEECH_DEVICE=cuda SPEECH_AUTH=on SPEECH_FAIL_FAST=true
bash "${CAS_ROOT}/scripts/runpod/run_speech_api.sh"
```

Script mặc định `SPEECH_HOST=127.0.0.1`, `SPEECH_PORT=8090`,
`SPEECH_WORKERS=1`, `SPEECH_MAX_INFLIGHT=1`. **Không mở cổng 8090 ra ngoài
pod.** `SPEECH_AUTH=off` chỉ được chấp nhận khi bind loopback.

## 3. Kiểm tra trên pod

```bash
curl -sS http://127.0.0.1:8090/healthz   # sống (không cần token)
curl -sS http://127.0.0.1:8090/readyz    # 200 sau khi ASR+MT load và qua mọi gate
```

`/readyz` chỉ 200 khi: MT contract hợp lệ, `tokenizer_parity=VERIFIED`,
`generation_matches_rq1=true` (256/256/4), checksum checkpoint đạt.
`research_provenance=NOT_VERIFIED` vẫn là ready hợp lệ và không bao giờ được
ghi đè thành `VERIFIED`. `init_ms` trong `/readyz` là thời gian load model,
không phải latency request.

Gọi infer thật (WAV PCM16 mono 16 kHz):

```bash
curl -sS -H "Authorization: Bearer ${SPEECH_SERVICE_TOKEN}" \
  -H "X-Request-ID: $(uuidgen)" -F "audio=@/path/to/bahnar.wav;type=audio/wav" \
  http://127.0.0.1:8090/v1/speech/infer
```

Sau nhiều request, `load_count` trong `/readyz` phải vẫn là `1` (không load lại
weight). `latency_ms` là thời gian ASR+MT; `duration_ms` là độ dài audio.

## 4. Dừng an toàn

Gửi SIGINT/SIGTERM cho process `python -m src.c0_speech`. Ngừng nhận ngay,
request đang chạy được chờ `SPEECH_SHUTDOWN_TIMEOUT_S` (mặc định 30). Không
dừng CUDA bằng cách xoá file checkpoint.

## 5. Docker local → RunPod (cách dùng hằng ngày)

Go chỉ chấp nhận `SPEECH_SERVICE_URL` loopback (`apps/api/internal/config/config.go`).
Container `speech-tunnel` (`deploy/speech-tunnel/`) chạy chung network namespace
với container `api` và giữ SSH forward `127.0.0.1:18090 → pod 127.0.0.1:8090`,
tự kết nối lại khi rớt.

1. Trên RunPod bật "SSH over exposed TCP" (`ssh root@<ip> -p <port>`). Proxy
   `ssh.runpod.io` không hỗ trợ port forwarding.
2. `cp deploy/.env.speech.example deploy/.env.speech.local` rồi điền token,
   `RUNPOD_SSH_HOST`, `RUNPOD_SSH_PORT`, `RUNPOD_SSH_KEY_PATH`. File `.local`
   bị gitignore. Pod mới đổi IP/port thì sửa file này và chạy lại `make up`.
3. `make up`. Kiểm tra:

```bash
cd deploy
docker inspect -f '{{.State.Health.Status}}' $(docker compose ps -q speech-tunnel)   # healthy
docker compose logs --tail=20 speech-tunnel
```

`unhealthy` nghĩa là tunnel rớt hoặc Speech trên pod không trả `/healthz`.
Khi đó route `/speech/translate` trả `SPEECH_UNAVAILABLE`; chat chữ vẫn chạy.

Debug thủ công từ Mac (không cần cho Docker): `ssh -N -L 18090:127.0.0.1:8090 root@<ip> -p <port>`
rồi `curl http://127.0.0.1:18090/healthz`.

## 6. Go và Speech cùng namespace

Khi Go chạy như process trên chính pod: `SPEECH_SERVICE_URL=http://127.0.0.1:8090`,
không cần tunnel. Container Docker `api` **không** dùng chung `127.0.0.1` với
host, nên `deploy/docker-compose.production.yml` (bản VPS) để `SPEECH_ENABLED=false`.

Audit speech là best-effort: lỗi INSERT không đổi HTTP 200 của bản dịch, được
log `speech_audit_failed`. Kiểm tra (chỉ đọc):

```sql
SELECT request_id, outcome, http_status, model_id, tokenizer_parity,
       research_provenance, chat_turn_request_id IS NOT NULL AS has_chat_link
FROM speech_translation_requests ORDER BY created_at DESC LIMIT 5;
```

## 7. Go-live toàn bộ hệ thống trên một GPU pod — KẾ HOẠCH

Chưa có code (`deploy/runpod/` chưa tồn tại). Các giả định dưới đây **chưa
được kiểm chứng trên pod thật** và phải kiểm trước khi làm.

Ràng buộc đã kiểm trong code:

| Ràng buộc | Nguồn | Hệ quả |
|---|---|---|
| `APP_ENV=prod` bắt buộc `sslmode=require` | `config.go` `validateProduction` | Postgres trên pod phải bật `ssl=on` (cert tự ký) |
| Prod từ chối object storage không TLS, `minioadmin`, auto-create bucket | `config.go`, `apps/ai-service/app/config.py` | Dùng S3 API của network volume RunPod thay MinIO |
| Speech URL chỉ loopback | `config.go` | Cùng pod: `http://127.0.0.1:8090` |
| Image phải `linux/amd64` | Mac build ra arm64 | `docker buildx --platform linux/amd64` |

Giả định cần kiểm chứng trước (UNVERIFIED):

- Pod GPU không chạy được Docker/Compose bên trong → các dịch vụ chạy thành
  process, quản lý bằng `supervisord`, trong một image.
- Chỉ `/workspace` còn sau restart → dữ liệu Postgres/Qdrant, model ở `/workspace/cas/`.
- Proxy HTTPS RunPod (`https://<podId>-<port>.proxy.runpod.net`) cắt request
  khoảng 100 giây → hạ `SPEECH_TIMEOUT_MS` (hiện 130000) xuống ≤ 90000.
- S3 API khả dụng cho network volume của tài khoản.

Kiến trúc dự kiến: một cổng HTTPS (nginx `:8088`) → web tĩnh + `/api/` →
Go `:8080`; Postgres `:5432`, Qdrant `:6333`, ai-service `:8001`, Speech `:8090`
chỉ bind `127.0.0.1`. Backup bằng `pg_dump` định kỳ lên S3 volume, có thử restore.
Chuyển dữ liệu local: `pg_dump`/`pg_restore`, `aws s3 sync` bucket `cas-documents`,
snapshot Qdrant `knowledge_chunks` (hoặc reindex).

Quyết định còn chờ: registry image, loại GPU/datacenter, mang dữ liệu local
hay khởi tạo mới, chấp nhận timeout Speech 90 giây.
