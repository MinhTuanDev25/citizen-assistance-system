# Exact Git delivery commands (DO NOT EXECUTE in this patch)

Chạy trên RunPod **chỉ sau khi** owner đã merge Phase 7 vào `main`.
Không chạy các lệnh này từ máy local của patch này.

Không dùng: `git reset --hard`, `git clean`, `git stash`,
`git checkout -- .`, hay bất kỳ lệnh nào sửa working tree hiện tại.

## 0. Điều kiện trước

- Phase 7 đã được commit trên `develop/citizen-assistance`.
- Owner đã merge vào `main` trên remote.
- Checkout `/workspace/citizen-assistance-system` vẫn dirty RQ2 — phải giữ.

## 1. Quan sát Git hiện tại (read-only)

```bash
cd /workspace/citizen-assistance-system
git rev-parse --is-inside-work-tree
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
git status --porcelain
git diff --stat
git remote
df -h /workspace
du -sh /workspace/citizen-assistance-system /workspace/bahnar-s2tt-thesis
test ! -e /workspace/citizen-assistance-phase7 && echo "DEST_FREE=YES" || echo "DEST_EXISTS=STOP"
```

Kỳ vọng (đã quan sát):

```text
branch = main
commit = e898c29fa7936fd94a87cd76aedcfa13b1dc8ba6
modified = bahnar-s2tt-thesis/notebooks/14_RQ2_Final_Train_Evaluate.ipynb
untracked = bahnar-s2tt-thesis/artifacts/rq2
untracked = bahnar-s2tt-thesis/notebooks/14_RQ2_Final_Train_Evaluate.ipynb.before_single_seed_backup
```

Nếu `DEST_EXISTS=STOP`: không tạo worktree, không xóa thư mục đích.

## 2. Fetch remote main (không đổi file working tree)

`git fetch origin main` không checkout và không xóa file dirty/untracked.

```bash
cd /workspace/citizen-assistance-system
git fetch origin main
git rev-parse origin/main
git log -1 --oneline origin/main
```

Ghi lại commit `origin/main`. Phải khác `e898c29…` sau khi merge Phase 7.

## 3. Kiểm tra file Phase 7 có trên remote main

```bash
cd /workspace/citizen-assistance-system
git cat-file -e origin/main:scripts/phase7/inspect_runpod_phase7.sh
git cat-file -e origin/main:scripts/phase7/inspect_runpod_lib.py
git cat-file -e origin/main:scripts/phase7/test_inspect_runpod.py
git cat-file -e origin/main:bahnar-s2tt-thesis/src/c0_cascaded.py
git cat-file -e origin/main:bahnar-s2tt-thesis/configs/c0_inference.example.yaml
git cat-file -e origin/main:bahnar-s2tt-thesis/src/c0_tokenizer.py
echo "PHASE7_FILES_ON_MAIN=YES"
```

Nếu `cat-file` fail: **STOP**. Chưa merge / chưa push. Không tạo worktree.

## 4. Disk trước khi tạo worktree

Worktree chỉ checkout source Git, không nhân bản checkpoint.

```bash
df -h /workspace
du -sh /workspace/bahnar-s2tt-thesis/bahnar_s2tt/full_state \
      /workspace/bahnar-s2tt-thesis/bahnar_s2tt/mt_full_state \
      /workspace/bahnar-s2tt-thesis/bahnar_s2tt/direct_full_state
```

Không `cp -a` các thư mục đó.

## 5. Tạo worktree tách biệt

Chỉ khi bước 1–4 đều OK và `/workspace/citizen-assistance-phase7` chưa tồn tại:

```bash
cd /workspace/citizen-assistance-system
git worktree add --detach /workspace/citizen-assistance-phase7 origin/main
```

Lệnh này không sửa file trong `/workspace/citizen-assistance-system`.

## 6. Xác nhận worktree + không nhân bản checkpoint

```bash
test -f /workspace/citizen-assistance-phase7/scripts/phase7/inspect_runpod_phase7.sh
test -f /workspace/citizen-assistance-phase7/bahnar-s2tt-thesis/src/c0_cascaded.py
test -f /workspace/citizen-assistance-phase7/bahnar-s2tt-thesis/configs/c0_inference.example.yaml
test ! -e /workspace/citizen-assistance-phase7/bahnar-s2tt-thesis/bahnar_s2tt/full_state
test -d /workspace/bahnar-s2tt-thesis/bahnar_s2tt/full_state
cd /workspace/citizen-assistance-system
git status --porcelain
git rev-parse HEAD
```

Checkout cũ vẫn phải là `e898c29…` (hoặc commit cũ) và vẫn dirty RQ2.

## 7. Inspection (không load model, không download)

```bash
export PHASE7_ARTIFACT_ROOT=/workspace/bahnar-s2tt-thesis
bash /workspace/citizen-assistance-phase7/scripts/phase7/inspect_runpod_phase7.sh
```

## 8. Tạo runtime YAML sau inspection (chưa chạy inference ở bước này trừ khi owner yêu cầu)

```bash
SRC=/workspace/citizen-assistance-phase7/bahnar-s2tt-thesis
cp "$SRC/configs/c0_inference.example.yaml" "$SRC/configs/c0_inference.runpod.yaml"
```

Điền đúng 3 path ARTIFACT_ROOT đã xác nhận (ASR `checkpoint-12500`,
MT `checkpoint-17500`, `mt_training_contract.json`). Tokenizer serving
vẫn là Hub pin `vinai/bartpho-syllable` @ `36eee8b4d648dd99da56462edcda3c5c97f7f3de`.

C0 inference **không** chạy trong patch này.
