# Switching the existing RTX 4090 pod to cas-aio (Phase 8.1-B)

This runbook moves the existing RunPod pod from its PyTorch research image to the all-in-one image, on the same network volume. It has three stages:

1. Prepare: check, preserve, apply. The pod keeps its old image.
2. Switch the image.
3. Verify, or roll back.

Nothing here trains a model or writes to the research tree. Every step that costs money, or that writes to the volume, waits for the operator.

## Facts this is built on

| | |
|---|---|
| Volume | `bahnar-thesis-storage`, ID `buppvu4d48`, 200 GB, US-IL-1, mounted at `/workspace` |
| CPU | AMD EPYC 7B13: AVX2 yes, AVX-512 VNNI no. E5 therefore runs **fp32** |
| GPU | NVIDIA RTX 4090, 24 GB |
| Image | `ghcr.io/minhtuandev25/citizen-assistance-system/cas-aio@sha256:40deae613fcc8f716d1874e41a12c565dfcd0e0e95b2ffb3fc4de396ca8cc566`, revision `b474deb`, amd64, 5.03 GB compressed. Torch 2.8.0 from PyPI, a CUDA 12.8 build; the C0 venv layer alone is 4.1 GB |
| ASR | `checkpoint-12500` (`full_xlsr300m_v1`), 5/5 required files. `model.safetensors` sha256 `dacbe7f82d8ed0a33f7e9798860d0e1808a20b9dc84725355a2db9b4229a6a2a` |
| MT | `checkpoint-17500` (`mt_bartpho_syllable_v1`), 6/6 required files. `model.safetensors` sha256 `f9bee0d53399f76548eee297fc67ff83bad4e43cd8fda630c9cb6dab93284ead` |
| MT contract | `.../mt_full_state/contract_c3d7b95698738634/mt_training_contract.json` |
| Tokenizer | `vinai/bartpho-syllable` @ `36eee8b4d648dd99da56462edcda3c5c97f7f3de`. This is the locked Hub pin, loaded offline. It is not the copy inside the MT checkpoint |

## Files

| File | Runs on | Role |
|---|---|---|
| `pod_stage.sh` | old pod, as root | `check` is read-only. `preserve` and `apply` write under `/workspace/cas` only |
| `c0_inference.yaml` | installed by `apply` | C0 config: absolute checkpoint and contract paths, `device: cuda`, the RQ1 generation contract, both weight pins, the tokenizer pin |
| `env.rtx4090.example` | RunPod "Edit Pod" | Environment, with secrets taken from RunPod Secrets |
| `verify_after_switch.sh` | new container, as root | Acceptance checks after the switch. Read-only |
| `rq_fingerprint.py` | both | Metadata listing of `/workspace`, excluding `/workspace/cas`: names, sizes and mtimes. It detects any write to the research data. Content is hashed only for the inference files, through the manifest |
| `tests/run_offline_tests.sh` | a machine with Docker | Offline tests in throwaway containers: `offline_harness.sh` (simulated pod on a mounted `/workspace`) and `nomount.sh` (no volume: nothing written). Prints `HARNESS:` and `NOMOUNT:` totals |

`CAS_STAGE_TEST_HW=<dir>` is for **tests only**: a fake `cpuinfo` and `nvidia-smi` in that directory stand in for the CPU and GPU. Every run that uses it prints `WARN simulated-hw` and writes `simulated_hardware=yes` into `READY_TO_SWITCH`, and `verify_after_switch.sh` then fails. Never set it on the pod.

### What `apply` writes

```
/workspace/cas/
  models/embedding/            e5 fp32, from fetch_models.sh (SHA-256 of every file)
  hf-cache/hub/models--vinai--bartpho-syllable/snapshots/36eee8b4…/   tokenizer only, no weights
  config/c0_inference.yaml     C0_INFERENCE_CONFIG
  manifests/c0_checkpoints.sha256
  preflight/                   verify_after_switch.sh, rq_fingerprint.py, rq_tree.listing.gz, rq_tree.fingerprint
  preserve/<pod>-<ts>/         only from `preserve`
```

The manifest has absolute paths. `entrypoint.sh` checks it in gpu mode with `cd / && sha256sum --strict -c`, and then confirms that the `speech` user (uid 10013) can read every file in it. The manifest lists:

- the 5 required ASR files, plus `special_tokens_map.json` and `added_tokens.json` when they are present (the ASR processor reads them);
- the 6 required MT files;
- the MT contract;
- every file of the tokenizer snapshot.

Training-state files (`optimizer.pt` and the like) are not in the manifest. `apply` refuses to write the manifest unless both weight hashes equal the pins above.

On the first boot the image itself creates `pgdata/`, `qdrant/`, `minio/`, `tls/`, `locks/` and `backups/` under `/workspace/cas`. The image writes nothing else on the volume. `ARTIFACT_ROOT` and the checkpoint paths are only read.

## Configuration in the new container

| Item | Value |
|---|---|
| E5 | `EMBEDDING_VARIANT=fp32`, model `/workspace/cas/models/embedding`. Boot log: `-> fp32 … avx512_vnni=no` |
| PostgreSQL 16 | `127.0.0.1:5432`, TLS `verify-full` with the internal CA, data in `pgdata/`. `POSTGRES_PASSWORD` is applied at the first init only |
| Qdrant 1.13.4 | `127.0.0.1:6333`, collection `knowledge_chunks` |
| MinIO | `https://127.0.0.1:9000`, bucket `cas-documents`, console off |
| Speech C0 | `127.0.0.1:8090`, `SPEECH_DEVICE=cuda`, Bearer `SPEECH_SERVICE_TOKEN`. Go reaches it on loopback |
| nginx | `:8088`, the only port that listens beyond loopback |
| Secrets (RunPod Secrets) | `CAS_JWT_SECRET` (≥ 32), `CAS_AI_SERVICE_TOKEN`, `CAS_SPEECH_SERVICE_TOKEN`, `CAS_POSTGRES_PASSWORD`, `CAS_MINIO_ROOT_PASSWORD` (≥ 16 each), `CAS_MINIO_ROOT_USER` (≥ 8, not containing `minioadmin`), `CAS_GEMINI_API_KEY`. Generate each with `openssl rand -hex 32`, and keep a copy outside RunPod |

## Procedure

Stop at the end of each step and read the output.

### 1. Record the pod as it is, for the rollback

From RunPod Console > Pod > Edit, write down:

- the exact container image, with its tag or digest;
- the container start command;
- the container disk size (GB) and the volume mount path;
- the HTTP and TCP ports;
- the environment variable names.

Without these values the rollback cannot be done exactly.

### 2. Check (read-only)

Finish or stop any RQ1/RQ2 job, Jupyter kernel or TensorBoard first. The switch kills every process in the old container.

Use the **full SHA of the reviewed commit**: the merge commit of the staging kit on `main`, which `git rev-parse origin/main` prints after the merge. Check out exactly that SHA, and give the same SHA to the script.

```bash
git clone --filter=blob:none https://github.com/MinhTuanDev25/citizen-assistance-system.git /tmp/cas-src
export CAS_STAGE_EXPECT_COMMIT=<full 40-character SHA of the reviewed commit>
git -C /tmp/cas-src checkout "$CAS_STAGE_EXPECT_COMMIT"
cd /tmp   # not inside /workspace: a shell there counts as activity on the volume
bash /tmp/cas-src/deploy/runpod/staging/pod_stage.sh check 2>&1 | tee /tmp/stage-check.txt; echo "exit=${PIPESTATUS[0]}"
```

You need **0 FAIL** and `exit=0`. The pipe into `tee` would hide the script's exit code, so `PIPESTATUS[0]` prints it (in bash; in zsh use `${pipestatus[1]}`). Treat these lines as decisions:

| Line | What it means |
|---|---|
| `WARN workspace-idle` | Something still uses `/workspace` |
| `WARN gpu-cuda` | The driver supports less than CUDA 12.8 |
| `INFO container-disk` / `keep-candidate` | Files that are lost at the switch |
| `volume-free` | The 200 GB quota minus the usage measured with `du`. `df` on a network volume shows the shared backend (hundreds of TB), not the quota, so it is printed as information only. The 200 GB is configured (`VOL_QUOTA_GB`), **not read from RunPod**. Confirming the size and usage in Console > Storage is a required manual step before `apply`. Any `du` error counts as FAIL, never as a partial number |

`apply` refuses to run unless `CAS_STAGE_EXPECT_COMMIT` is set and equals the clone's HEAD (`source-commit`). `check` only warns when it is unset. Together with `source-clean`, this proves that the code that runs is the reviewed commit, byte for byte. `READY_TO_SWITCH` records that commit as `clone=`.

`check` also requires `/workspace` to be a mount point (`volume-mount`). Without the network volume, everything would be prepared on the container disk and lost at the switch.

The clone must be a blob-less clone, not `--depth 1`, because `check` compares the C0 and E5 files with image revision `b474deb`.

`check` (and `apply`) also refuse two more situations:

- **A clone whose working tree differs from its commit** (`source-clean`). This covers modified, staged, untracked or ignored files in the C0 code, `fetch_models.sh`, the e5 manifests, `embed.py` and `staging/`. The code that runs must be exactly the reviewed commit; if it is not, clone again.
- **Unsafe paths** (`paths`). `/workspace` or `/workspace/cas` is a symlink, a directory the scripts write to is a symlink, or a symlink under `/workspace/cas` resolves outside it. Any of these could lead a write into the research tree.

### 3. Preserve what lives only on the container disk

Use the `container-disk` and `keep-candidate` lines from step 2, for example:

```bash
bash /tmp/cas-src/deploy/runpod/staging/pod_stage.sh preserve /root/.jupyter /root/.config
```

`preserve` writes to `/workspace/cas/preserve/<pod>-<ts>/`:

- pip freeze;
- `nvidia-smi -q`;
- environment variable **names**;
- the process list and crontab;
- the list of files changed on the container disk;
- a tar of the paths you pass.

It refuses paths that are already on the volume, because those survive the switch. It also refuses to write anything when `/workspace` is not a mount point or a path under `/workspace/cas` is unsafe; a backup on the container disk would be lost at the switch.

Anything not preserved is lost: `pip install`s, `~/.cache/huggingface`, shell history, SSH `authorized_keys`, notebooks saved outside `/workspace`. If the tar contains secrets, move it off RunPod or keep the directory at `0700`.

### 4. Apply (writes under `/workspace/cas` only)

```bash
cd /tmp
# same shell as step 2: CAS_STAGE_EXPECT_COMMIT must still be exported
bash /tmp/cas-src/deploy/runpod/staging/pod_stage.sh apply 2>&1 | tee /tmp/stage-apply.txt; echo "exit=${PIPESTATUS[0]}"
```

`apply` first removes any old `preflight/READY_TO_SWITCH`. It then runs every check of `check` again and writes nothing if any of them fails. Those checks cover the CPU (AVX2), the GPU and the driver, path safety, a clean clone, the volume quota, the checkpoints and pins, and read access. It then runs these steps in order, and **stops at the first step that fails**; no later step writes anything:

1. **Probes the filesystem.** It runs what the first boot will do: `chown` to uid 10010, mode `0700`, `fsync`, rename, `flock`, `mkdir`, and an `O_DIRECT` test.
2. **Installs the E5 model.** It runs `fetch_models.sh --variant fp32`, which downloads about 0.5 GB from the Hub.
3. **Fills the tokenizer cache.** It installs `transformers==4.57.6` in `/tmp/cas-c0-tok`, then downloads the pinned tokenizer files without weights.
4. **Checks the Speech serving gate.** It runs `python -m src.c0_tokenizer --contract …` **as uid 10013 with `HF_HUB_OFFLINE=1`**. The result must be `VERIFIED`.
5. **Installs `c0_inference.yaml`.**
6. **Writes and verifies the manifest.**
7. **Saves the RQ fingerprint**, the metadata of everything under `/workspace` except `/workspace/cas`.

Only when every step passed does `apply` write `/workspace/cas/preflight/READY_TO_SWITCH`. The marker records the time, the clone commit, the image revision, the SHA-256 of the manifest and of the config, and the number of WARN. A run that stops prints `STOPPED at step '…'. NOT ready to switch`.

Running it again is safe: completed steps are checked again and kept, and the run continues from the step that failed. A file that already exists and differs is never overwritten: `apply` reports FAIL and stops.

Do not start RQ work between `apply` and the switch. Any change in `/workspace` shows up as a difference in step 8.

### 5. Secrets

Create the seven RunPod Secrets listed above. RunPod Secrets are free.

### 6. Access and recovery: check before the switch

**Recovery never needs a shell.** The rollback below is an edit in RunPod Console (the control plane). It brings back the old image and leaves `/workspace` as it was. It works even if the new container crash-loops.

**The shell after the switch:** RunPod's basic SSH goes through its proxy (`ssh <pod>@ssh.runpod.io -i <key>`). RunPod documents it as working with any container, without `sshd` and without exposed ports, using the SSH public key stored in the account. It cannot copy files (no SCP). Whether it works with this image is **UNVERIFIED** until the switch. Before the switch:

1. In RunPod Settings > SSH Public Keys, check that your public key is there.
2. Connect to the current pod with the basic SSH command from Connect. This checks the account and key path at no extra cost.
3. Do not open a public SSH port and do not add `sshd` for this.

Right after the switch there is a **go/no-go check, 10 minutes at most**: open the basic SSH session. If there is no shell, roll back at once. During a `FATAL` crash loop no container stays up, so no shell is possible; the reason is in the Logs tab.

### 7. Switch the image on the same pod

Switch only if the last `apply` printed `exit=0` and `READY_TO_SWITCH written`, and `cat /workspace/cas/preflight/READY_TO_SWITCH` shows that run's time.

In RunPod Console > Pod > Edit:

- **Container image:** the digest reference from "Facts".
- **Container start command:** empty.
- **Container disk:** keep the current size if it is ≥ 30 GB.
- **Volume mount path:** `/workspace`, unchanged.
- **Environment:** the variables from `env.rtx4090.example`.
- **HTTP ports:** **do not add `8088`.** The RunPod proxy URL is public, and opening it is a separate decision. Keep the old ports as they are; nothing listens on them in the new image.

Saving recreates the container. The container disk is wiped, and `/workspace` is kept. The pod ID stays the same, so the single-writer lease owner is `runpod-<pod id>`.

### 8. Verify

**In the Logs tab, within about 10 minutes:**

- `speech_mode=gpu`
- `embedding model verified … -> fp32 … avx512_vnni=no`
- `gpu mode: verifying checkpoint manifest (N files…)`
- `gpu mode checks passed: GPU 0: NVIDIA GeForce RTX 4090`
- `bootstrap` reaches `bootstrap.ok`
- no `FATAL` and no `MAINTENANCE` line

**In the pod shell:**

```bash
# --audio is required: one real C0 inference on a sample the research data already holds (read-only)
bash /workspace/cas/preflight/verify_after_switch.sh --audio /workspace/bahnar-s2tt-thesis/<path to a test WAV>; echo "exit=$?"
```

You need **0 FAIL** and `exit=0`. The script checks:

- that `READY_TO_SWITCH` exists, says `simulated_hardware=no`, and still matches: its `image_revision` equals the running image, its manifest and config SHA-256 equal the current files, and Speech uses that config;
- `healthcheck.sh`, with Speech required;
- `torch.cuda.is_available()` in the C0 venv;
- Speech `/readyz`, which must show `ready`, `tokenizer_parity=VERIFIED` and `generation_matches_rq1=true`;
- that the models are on the GPU. Speech is ready although it is forced to `SPEECH_DEVICE=cuda` (it fails closed without CUDA), and the speech process itself holds `/dev/nvidia*` open. VRAM in use is not taken as proof;
- one real C0 inference (`--audio`): HTTP 200, non-empty Bahnar and Vietnamese text, steps 12500/17500, `configured_hash_match`, latency;
- E5: `EMBEDDING_VARIANT` must be `fp32` (anything else is a FAIL) and `EMBEDDING_CPU_FLAGS` must be unset. One real embedding then runs with the ai-service's own class, as user `cas`, on the real CPU: 1 query and 16 passages, 384 dimensions, finite, unit norm. The script reports load, query and batch latency and the peak RSS;
- Gemini, as a **required** gate (extraction and Definition drafts depend on it). The provider must be `gemini` and a key must be configured. The script then takes both effective model IDs from ai-service `/v1/status` and calls `models.get` for each, which is metadata only and bills no tokens. Only HTTP 200 passes; a missing model, a refused key or no answer is a FAIL. A Draft job remains the end-to-end test;
- that only `:8088` listens beyond loopback. Any other listener is a FAIL that names its process; if RunPod injects one (for example for its terminal), identify it before accepting;
- migrations clean;
- data counts, read-only;
- that the research tree matches the fingerprint from `apply`. The fingerprint compares names, sizes and mtimes, not content: it detects writes, but it is not a byte-level proof for the whole tree;
- that the bytes of every inference file are unchanged. `sha256sum -c` of the manifest re-hashes the weights, configs, contract and tokenizer, about 3 GB.

On a first boot the database holds only what the migrations seed: in a local test, 1 user and 5 procedures, with 0 documents and 0 chunks. That is expected; see blocker 1.

`checkpoint_identity` will be `configured_hash_match` and `research_provenance` will be `NOT_VERIFIED`. That is by design: `configs/c0_research_pins.yaml` stays empty, and filling it is a research decision.

## Rollback

Roll back when any of these happens:

- the Logs tab shows `[entrypoint] … FATAL` (exit 64, container restarting) or `MAINTENANCE`;
- `verify_after_switch.sh` still has a FAIL 30 minutes after the switch;
- there is no shell 10 minutes after the switch (step 6).

To roll back:

1. RunPod Console > Pod > Edit:
   - set the image, start command, disk, ports and environment back to the values recorded in step 1;
   - remove the CAS variables and the secret references;
   - Save.

   The image's SIGTERM stops the services cleanly and releases the lease.
2. Restore anything you preserved: `tar -xzf /workspace/cas/preserve/<dir>/container-files.tar.gz -C /`, then reinstall from `pip-freeze.txt` if needed.
3. Confirm the research data is unchanged:

   ```bash
   python3 /workspace/cas/preflight/rq_fingerprint.py /workspace --exclude /workspace/cas \
     --compare /workspace/cas/preflight/rq_tree.listing.gz      # IDENTICAL
   sha256sum <ASR dir>/model.safetensors <MT dir>/model.safetensors   # the two pins
   ```
4. Leave `/workspace/cas` in place. It holds only the new stack's data. Deleting it is a separate, explicit decision.

To try the switch again later, run `apply` again (it must write `READY_TO_SWITCH`), then repeat step 7. `/workspace/cas` is reused, including `pgdata` and the `POSTGRES_PASSWORD` it was initialised with.

## Blockers and unverified points

1. **Production data.** The first boot creates a new Postgres (migration seed only), an empty Qdrant collection and an empty MinIO. The current data lives in the Mac Docker stack. Moving it needs its own approved plan: `pg_dump`/restore, the MinIO objects, and a reindex (or a Qdrant snapshot), followed by `bootstrap-admin`. Activation stays manual.
2. **Shell access after the switch is UNVERIFIED.** The image has no `sshd`. RunPod documents basic proxied SSH for any container, but that has not been tried on this image. Step 6 limits the risk: the key path is checked beforehand, there is a 10-minute go/no-go, and the rollback is a Console edit that needs no shell. No public port and no `sshd` are added.
3. **Public access is not opened.** Exposing `8088` gives a public proxy URL. The client IP header and the 100 s proxy limit are unverified (see `../README.md`, "Known limits").
4. **Network volume behaviour.** `apply` probes `chown`, modes, `fsync`, `flock` and `O_DIRECT` once. Postgres durability and performance on the network volume, and `flock` between two pods, stay unverified. Run only one cas-aio container on this volume.
5. **CUDA.** cu128 torch needs a driver with CUDA 12.8 support, or else it relies on minor-version compatibility. `check` reports the driver, and `verify_after_switch.sh` decides.
6. **Research environment unavailable.** While cas-aio runs, the pod is not the PyTorch research environment. RQ work needs the rollback, or another pod on another volume (to keep a single writer).
7. **Cost.** As production, the GPU pod runs around the clock, billed hourly. The prices are NOT_VERIFIED; check them in the console. `backup.sh` writes to the same volume, so off-RunPod backups are still to do.
8. **Pod mobility.** After a stop, the pod may start on another host of the data center. fp32 E5 keeps working there, but GPU availability is not guaranteed.
