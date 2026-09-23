# Bahnar → Vietnamese S2TT (RQ1)

Master's thesis workspace: compare **cascaded ASR + MT (C0)** with **direct speech-to-text translation (D0)** on `cuong06/Bahnar_Vietnamese`.

The citizen-assistance chatbot lives outside this folder and is not part of RQ1. Frozen test remains closed until Notebook 06.

## Layout

```text
bahnar-s2tt-thesis/
├── notebooks/          # 01 audit → 06 evaluation
├── src/                # ASR / MT / Direct helpers
├── configs/            # asr.yaml, mt.yaml, direct.yaml
├── tests/
├── data/manifests/     # locked RQ1 splits from Notebook 01
├── data/audit/
├── artifacts/          # exported bundles (not required to clone)
└── requirements.txt
```

## Notebooks

| # | File | What it does |
|---|------|----------------|
| 01 | `notebooks/01_data_audit_and_split.ipynb` | Audit HF rows, lock `recording_group_id` splits, write `rq1_*.csv` |
| 02 | `notebooks/02_asr_data_preflight_and_smoke_test.ipynb` | Audio/PCM preflight before full ASR |
| 03 | `notebooks/03_asr_baseline_training.ipynb` | Cascaded ASR: XLS-R-300m CTC, Bahnar speech → Bahnar text |
| 04 | `notebooks/04_mt_baseline_training.ipynb` | Cascaded MT: BARTpho-syllable, Bahnar text → Vietnamese |
| 05 | `notebooks/05_train_direct_s2tt.ipynb` | Direct D0: XLS-R-300m encoder + mBART-50-mmt `vi_VN` decoder |
| 06 | `notebooks/06_rq1_evaluation.ipynb` | Final C0 vs D0 on frozen test (`verify → unlock_test → run_cascaded → run_direct → finalize`) |

Matched D0 speech UIDs follow the NB03 eligible set (currently **102,486** train / **11,112** validation). D0 joins only locked Vietnamese targets from RQ1 manifests; Bahnar text is not a Direct model input.

## Environment

Do **not** install into a shared Anaconda/base env.

```bash
cd bahnar-s2tt-thesis
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name=bahnar-s2tt --display-name="Python (bahnar-s2tt)"
```

Pinned training runtime (Notebook 03/04/05/06):

- `torch==2.8.0`
- `transformers==4.57.6`
- `accelerate==1.10.1`

In Jupyter: **Select Kernel → Python (bahnar-s2tt)**. Static check without running notebooks:

```bash
python -m pytest -q
```

## How to run stages

Every full notebook uses `FULL_STAGE` in order. Do not jump to `train`.

```text
prepare → pilot → resume_test_a → restart kernel → resume_test_b → train → evaluate
```

- Keep `ALLOW_FULL_TRAINING=False` until `FULL_STAGE="train"`.
- Set `BAHNAR_DURABLE_CHECKPOINT_BUDGET_BYTES` before `resume_test_*` / `train` (no notebook default).
- Default RunPod paths: local `/tmp/bahnar-runtime`, durable `/workspace/bahnar-s2tt-thesis`.
- `HF_HUB_ENABLE_HF_TRANSFER=0`.

### Notebook 05 extra gate (before prepare)

`configs/direct.yaml` `nb03` pins must be filled from the **real** NB03 eligible CSVs:

- `train_eligible_uid_set_hash`
- `validation_eligible_uid_set_hash`
- `train_eligible_file_sha256`
- `validation_eligible_file_sha256`

Empty pins fail closed and print the hashes computed from the current CSVs. Paste those four values, restart the kernel (YAML changes the source fingerprint), then prepare again. Do not invent hashes. Changing YAML after a prepare invalidates the training contract.

Locked Direct models:

- Encoder `facebook/wav2vec2-xls-r-300m` @ `1a640f32ac3e39899438a2931f9924c02f080a54`
- Decoder `facebook/mbart-large-50-many-to-many-mmt` @ `1fc5c3d1fc340141fe0daf7b9898d85c4e60b436`, `target_lang=vi_VN`
- `MAX_AUDIO_DURATION = 40`

Local checkpoint peak is `save_total_limit + 1` (default 3). Durable peak is 4.

## Start here

1. Open Notebook 01 with kernel `bahnar-s2tt`.
2. Keep `GROUP_REVIEW_APPROVED=False` on the first audit pass.
3. Review `data/audit/`, then approve the split.
4. Continue 02 → 03 → 04, then 05 on RunPod after NB03 eligible CSVs exist.
5. Notebook 06 opens only after NB03/NB04/NB05 all have durable best + evaluate success (`SUCCESS_*_EVALUATE`) and frozen-test clean. Stages: `verify → unlock_test → run_cascaded → run_direct → finalize`. Fill exact `ASR_STATE_DIR` / `MT_STATE_DIR` / `DIRECT_STATE_DIR` in the operator-control cell (no auto-latest).
