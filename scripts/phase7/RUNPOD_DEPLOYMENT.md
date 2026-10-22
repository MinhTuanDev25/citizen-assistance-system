# Phase 7 Step 7.3 — RunPod deployment runbook

Không commit, merge, push, pull, reset, clean, stash, hay chạy inference
trong tài liệu này. Các lệnh dưới đây chỉ chạy **sau khi owner đã review
patch, commit trên `develop/citizen-assistance`, và merge vào `main`.**

## Hai thư mục tách biệt

| Vai trò | Đường dẫn RunPod đã quan sát |
|---|---|
| CAS Git checkout hiện tại (giữ nguyên, có RQ2 dirty) | `/workspace/citizen-assistance-system` |
| Worktree Phase 7 đề xuất (sau khi merge `main`) | `/workspace/citizen-assistance-phase7` |
| THESIS_SOURCE_ROOT (code C0 + YAML mẫu) | `$CAS_ROOT/bahnar-s2tt-thesis` |
| ARTIFACT_ROOT (checkpoint đã train) | `/workspace/bahnar-s2tt-thesis` |

Không giả định checkpoint nằm trong THESIS_SOURCE_ROOT.
Không copy / rename / di chuyển checkpoint.

## Trạng thái Git hiện tại trên pod (phải giữ)

- Branch: `main`
- Commit: `e898c29fa7936fd94a87cd76aedcfa13b1dc8ba6`
- Modified: `bahnar-s2tt-thesis/notebooks/14_RQ2_Final_Train_Evaluate.ipynb`
- Untracked: `bahnar-s2tt-thesis/artifacts/rq2`
- Untracked: `bahnar-s2tt-thesis/notebooks/14_RQ2_Final_Train_Evaluate.ipynb.before_single_seed_backup`

Các thay đổi RQ2 **bắt buộc giữ**. Checkout hiện tại không được overwrite.

## Chính sách delivery

1. Owner review + commit trên nhánh phát triển hiện có `develop/citizen-assistance`.
2. Owner merge các thay đổi Phase 7 đã duyệt vào `main`.
3. RunPod lấy `main` đã duyệt bằng **git worktree mới**, không đụng working tree cũ.

Không tạo nhánh phát triển mới.
Không tự động merge / commit / push / pull từ script inspection.

## Checkpoint đã xác nhận tồn tại (file presence, chưa load)

ASR:

`/workspace/bahnar-s2tt-thesis/bahnar_s2tt/full_state/contract_b5172ee0cc391e4f/checkpoints/full_train/full_xlsr300m_v1/ckpts/checkpoint-12500`

- `model.safetensors`
- `config.json`
- `vocab.json`
- `tokenizer_config.json`
- `preprocessor_config.json`

MT:

`/workspace/bahnar-s2tt-thesis/bahnar_s2tt/mt_full_state/contract_c3d7b95698738634/checkpoints/full_train/mt_bartpho_syllable_v1/ckpts/checkpoint-17500`

- `model.safetensors`
- `config.json`
- `tokenizer_config.json`
- `sentencepiece.bpe.model`
- `dict.txt`
- `generation_config.json`

MT contract:

`/workspace/bahnar-s2tt-thesis/bahnar_s2tt/mt_full_state/contract_c3d7b95698738634/mt_training_contract.json`

RQ2 artifact root:

`/workspace/bahnar-s2tt-thesis/bahnar_s2tt/direct_full_state`

Serving tokenizer (tách biệt checkpoint MT): `vinai/bartpho-syllable`
revision `36eee8b4d648dd99da56462edcda3c5c97f7f3de`.

## Inspection sau khi có worktree

```bash
export PHASE7_ARTIFACT_ROOT=/workspace/bahnar-s2tt-thesis
bash /workspace/citizen-assistance-phase7/scripts/phase7/inspect_runpod_phase7.sh
```

Script resolve `CAS_ROOT` từ `BASH_SOURCE`. Không `cd` vào ARTIFACT_ROOT.

## Tạo runtime YAML (chưa tồn tại)

Mẫu thật sự:

`$CAS_ROOT/bahnar-s2tt-thesis/configs/c0_inference.example.yaml`

Không invent tên file mẫu khác. Copy sau khi source đã nằm trên worktree:

```bash
SRC=/workspace/citizen-assistance-phase7/bahnar-s2tt-thesis
cp "$SRC/configs/c0_inference.example.yaml" "$SRC/configs/c0_inference.runpod.yaml"
```

Điền đường dẫn đã xác nhận (ARTIFACT_ROOT, không phải THESIS_SOURCE_ROOT):

```yaml
asr_checkpoint: /workspace/bahnar-s2tt-thesis/bahnar_s2tt/full_state/contract_b5172ee0cc391e4f/checkpoints/full_train/full_xlsr300m_v1/ckpts/checkpoint-12500
mt_checkpoint: /workspace/bahnar-s2tt-thesis/bahnar_s2tt/mt_full_state/contract_c3d7b95698738634/checkpoints/full_train/mt_bartpho_syllable_v1/ckpts/checkpoint-17500
mt_training_contract: /workspace/bahnar-s2tt-thesis/bahnar_s2tt/mt_full_state/contract_c3d7b95698738634/mt_training_contract.json
device: cuda
```

Không commit file runtime nếu chứa secret hoặc hash máy cục bộ không cố ý.

## Lệnh cấm

- `git reset --hard`
- `git clean`
- automatic `git stash`
- overwrite `/workspace/citizen-assistance-system`
- copy thư mục checkpoint lớn sang worktree
- download Hub tokenizer trong inspection
- load ASR/MT trong inspection
