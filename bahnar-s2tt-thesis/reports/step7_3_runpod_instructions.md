# Step 7.3 — RunPod real-audio instructions

Do not start while another job holds the 4090. Confirm with `nvidia-smi`
that VRAM is free. Do not kill Jupyter/RQ2 from this guide.

These commands match the source after the provenance final fix. They are
not evidence that a run already happened.

RunPod has two separate trees. Do not collapse them:

```text
CAS_ROOT / Git worktree   /workspace/citizen-assistance-phase7
THESIS_SOURCE_ROOT        /workspace/citizen-assistance-phase7/bahnar-s2tt-thesis
ARTIFACT_ROOT             /workspace/bahnar-s2tt-thesis
```

Git delivery uses a **new worktree**. Do not `git reset --hard`, `git clean`,
or stash the existing `/workspace/citizen-assistance-system` checkout (it
holds uncommitted RQ2 notebook/artifacts). Exact commands:
`reports/phase7_step73_final_patch/GIT_DELIVERY_COMMANDS.md`.

## 1. Sync source onto the pod

Obtain Phase 7 source from approved `main` in
`/workspace/citizen-assistance-phase7` (see the Git delivery runbook).
C0 entrypoints live under THESIS_SOURCE_ROOT:

```text
src/c0_cascaded.py
src/c0_manifest.py
src/c0_tokenizer.py
configs/c0_inference.example.yaml
configs/c0_research_pins.yaml
```

Trained checkpoints stay at ARTIFACT_ROOT. Do not copy or rename them.

```bash
cd /workspace/citizen-assistance-phase7/bahnar-s2tt-thesis
python -c "import src.c0_cascaded, src.c0_tokenizer, src.c0_manifest"
```

Do not point inference at `/tmp/bahnar-runtime`.

## 2. ASR best checkpoint (`checkpoint-12500`)

Verified best step from ASR `trainer_state.json` (do not pick a later step):

```text
bahnar_s2tt/full_state/contract_b5172ee0cc391e4f/checkpoints/full_train/full_xlsr300m_v1/ckpts/checkpoint-12500
```

Example YAML field:

```yaml
asr_checkpoint: /workspace/bahnar-s2tt-thesis/bahnar_s2tt/full_state/contract_b5172ee0cc391e4f/checkpoints/full_train/full_xlsr300m_v1/ckpts/checkpoint-12500
```

Confirm the basename is exactly `checkpoint-12500` and that
`model.safetensors`, `config.json`, `vocab.json`, `tokenizer_config.json`,
and `preprocessor_config.json` are files in that directory.

## 3. MT best checkpoint (`checkpoint-17500`)

Verified best step from MT `trainer_state.json` (do not pick a later step):

```text
bahnar_s2tt/mt_full_state/contract_c3d7b95698738634/checkpoints/full_train/mt_bartpho_syllable_v1/ckpts/checkpoint-17500
```

```yaml
mt_checkpoint: /workspace/bahnar-s2tt-thesis/bahnar_s2tt/mt_full_state/contract_c3d7b95698738634/checkpoints/full_train/mt_bartpho_syllable_v1/ckpts/checkpoint-17500
```

Required files: `model.safetensors`, `config.json`, `tokenizer_config.json`,
`sentencepiece.bpe.model`, `dict.txt`, `generation_config.json`.

Ignore `best_model_checkpoint` absolute paths that still mention
`/tmp/bahnar-runtime`. Use the directories above.

## 4. Collect research-reference hashes

The paths in sections 2–3 are **where to look**, not proof of identity.

There is no locked GB-weight SHA in this repository. Do not invent one.
Hashing the files on the pod and writing that hash into YAML is a
**configured reference**, not independent research attestation.

On the pod, hash the research `model.safetensors` files once:

```bash
cd /workspace/citizen-assistance-phase7/bahnar-s2tt-thesis
python - <<'PY'
from pathlib import Path
from src.rq1_contract import sha256_file
root = Path("/workspace/bahnar-s2tt-thesis")  # ARTIFACT_ROOT, not Git source
asr = root / "bahnar_s2tt/full_state/contract_b5172ee0cc391e4f/checkpoints/full_train/full_xlsr300m_v1/ckpts/checkpoint-12500/model.safetensors"
mt = root / "bahnar_s2tt/mt_full_state/contract_c3d7b95698738634/checkpoints/full_train/mt_bartpho_syllable_v1/ckpts/checkpoint-17500/model.safetensors"
print("asr", sha256_file(asr))
print("mt", sha256_file(mt))
PY
```

- Put those hashes in `configs/c0_research_pins.yaml` if you want a
  reference-integrity check. Serving may then report
  `checkpoint_identity=research_reference_hash_match` and
  `research_provenance=REFERENCE_HASH_MATCH`. That is **not** `VERIFIED`.
- Leave the file empty to keep `research_provenance=NOT_VERIFIED`.
- You may still run inference. Successful inference is not provenance.

Never paste the live hash into `asr_weights_sha256` and treat that as
research verification. Operator YAML pins are integrity only.
`research_pins` is also a configured pointer, not a trust anchor.

## 5. Checksum integrity (operator pin)

Copy `bahnar-s2tt-thesis/configs/c0_inference.example.yaml` (in the CAS
worktree) to `configs/c0_inference.runpod.yaml` in the same directory.
Do not invent a different example filename. Checkpoint fields must use
ARTIFACT_ROOT paths from sections 2–3.

Optional but recommended: set operator pins to the hashes you just
computed so a later file swap fails closed:

```yaml
asr_weights_sha256: "<hex from section 4>"
mt_weights_sha256: "<hex from section 4>"
research_pins: configs/c0_research_pins.yaml
```

- Matching operator pin → `configured_hash_match` (`research_provenance`
  stays `NOT_VERIFIED` unless research-reference pins also match, in
  which case it becomes `REFERENCE_HASH_MATCH`, never `VERIFIED`).
- Mismatch → JSON `error_code=checkpoint_checksum_mismatch`, exit 1,
  models are not loaded.
- Empty operator pins → skip the integrity gate.

Quick local check without GPU load (manifest only):

```bash
python -m src.c0_manifest --config configs/c0_inference.runpod.yaml --out manifests/c0_manifest.runpod.json
```

Do not pass `--compute-sha256` unless you intend to hash the GB files again.

## 6. MT tokenizer pinned to RQ1

```yaml
mt_training_contract: /workspace/bahnar-s2tt-thesis/bahnar_s2tt/mt_full_state/contract_c3d7b95698738634/mt_training_contract.json
```

Adjust if the contract file lives next to the MT state under a different
name. Then:

```bash
python -m src.c0_tokenizer --contract /path/to/mt_training_contract.json
```

PASS only if JSON has `"tokenizer_parity": "VERIFIED"` and exit code 0.
If the contract is absent, omit the field and keep
`TOKENIZER_PARITY=NOT_VERIFIED`. Do not report tokenizer PASS.

Optional later check (canonical/local, **not** the stored RQ1 hash):

```bash
python -m src.c0_tokenizer --contract /path/to/mt_training_contract.json \
  --local-tokenizer /path/to/checkpoint-17500
```

## 7. Run real Bahnar audio

Required serving YAML (plus checkpoint paths from sections 2–3):

```yaml
device: cuda
max_source_length: 256
generation_max_length: 256
num_beams: 4
max_audio_duration_seconds: 30
max_audio_bytes: 10485760
```

Use a real 16 kHz Bahnar wav already on the pod. Other rates are rejected.
C0 does not resample.

```bash
nvidia-smi   # VRAM must be free
cd /workspace/citizen-assistance-phase7/bahnar-s2tt-thesis
python -m src.c0_cascaded --config configs/c0_inference.runpod.yaml --audio /path/to/bahnar.wav
```

## 8. Read the JSON

Keep stdout. On success, record:

- `asr_text_bahnar` — Bahnar transcript
- `translated_text_vi` — Vietnamese translation
- `duration_ms` — audio length
- `latency_ms` — wall time for this utterance
- `checkpoint_identity`
- `research_provenance`
- `tokenizer_parity`
- `generation_matches_rq1`

On failure, record `error` / `error_code` and the exit code (nonzero).

## 9. PASS / FAIL

| Check | PASS |
|---|---|
| Tokenizer | `python -m src.c0_tokenizer --contract ...` → `VERIFIED` and exit 0 |
| Integrity | Declared YAML SHA matches, or pins left empty on purpose |
| Provenance | Honest labels only: `NOT_VERIFIED` or `REFERENCE_HASH_MATCH`. `verified_c0` / `research_provenance=VERIFIED` is FAIL |
| Load | CLI starts, both models on CUDA, no OOM |
| Audio | 16 kHz wav, duration ≤ serving cap |
| Output | Non-empty Bahnar + Vietnamese, no `error` |
| Generation | `generation_matches_rq1=true` when using 256/256/4 |

Any `error_code`, traceback without JSON, invented translation, or
`verified_c0` / `research_provenance=VERIFIED` is FAIL. Reference-hash
match is allowed; it is not independent research provenance.

Dependencies: `torch==2.8.0`, `transformers==4.57.6`, `accelerate==1.10.1`
from `requirements-c0-inference.txt`. Record a mismatch as a residual;
do not silently ignore it.

## Common errors

| Symptom | What to check |
|---|---|
| `checkpoint_checksum_mismatch` | Operator YAML SHA ≠ file; do not empty the pin to “make it pass” unless you are choosing unverified mode |
| `checkpoint ... is incomplete` | Required files in that checkpoint directory |
| `fingerprint mismatch` | Contract is MT training contract; tokenizer is Hub pin |
| `sample rate != 16000` | Convert offline; C0 does not resample |
| `empty_asr` / `empty_mt` | Serving gate; try another utterance |
| `dependency_missing` / `model_load_failed` | JSON stdout plus stderr; VRAM and package versions |
| CUDA OOM | Another PID; 4090 cannot hold RQ2 and C0 together |

WebM/Opus from the browser is out of scope until the Speech API step.
