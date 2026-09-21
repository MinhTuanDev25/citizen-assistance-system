# Notebook 05 Direct S2TT — Implementation Report

Date: 2026-09-21 (follow-up: split monitor semantic vs byte SHA; require 4/4 NB03 pins)

This change closes the two remaining HIGH review items. **No prepare, pilot, resume-test, training, evaluate, or model download was executed.** Verification was limited to `pytest`, `compileall`, and AST compile of notebook code cells.

**READY_FOR_RUNPOD_PREPARE = False.** `configs/direct.yaml` still has empty NB03 eligible pins. Production counts `102486` / `11112` fail closed until all four hashes are pasted from the real NB03 CSVs on RunPod. Do not invent hashes.

## Spec gates closed

- Monitor semantic lock is `n` / `uid_set_hash` / `pair_hash` / `ordered_row_hash` vs `fixed_subset(validation)`. Pandas float/NaN/duration CSV round-trip no longer false-fails.
- Monitor byte lock is `sha256_file(actual monitor path)` vs `training_contract["monitor_file_sha256"]`. Persist writes the monitor CSV first, hashes the staged file, rebuilds the training contract with that SHA, verifies after write, then commits.
- Production NB03 counts require all four pins non-empty (`train`/`validation` UID set + file SHA-256), then verify those pins against the actual eligible CSVs before the row-count check. 0/4–3/4 fail; 4/4 matching hashes pass pin verification; one wrong hash fails on mismatch.
- YAML pins stay empty in this overlay. Eligible CSVs are not in the repo; hashes are not invented.
- `source_fingerprint_sha256` remains a training-contract field. `mt_contract.py` is not on the NB05 import path and is not added to the fingerprint.

## Unchanged scientific / runtime pins

- Encoder: `facebook/wav2vec2-xls-r-300m` @ `1a640f32ac3e39899438a2931f9924c02f080a54`
- Decoder: `facebook/mbart-large-50-many-to-many-mmt` @ `1fc5c3d1fc340141fe0daf7b9898d85c4e60b436`
- Target language: `vi_VN`
- `MAX_AUDIO_DURATION = 40`
- Local checkpoint peak = `save_total_limit + 1` = 3; durable peak = 4
- Runtime: `torch==2.8.0`, `transformers==4.57.6`, `accelerate==1.10.1`
- Resume uses SeedableRandomSampler; evaluate uses full `text_vi_norm`; validation truncation = 0
- Fail-closed `SUCCESS_DIRECT_TRAINING`; best durable checkpoint is validated before sync

## Files changed

- `src/direct_full_train.py` — semantic monitor assert; `assert_monitor_file_sha256`
- `src/direct_data.py` — persist staged-monitor SHA lock; production 4/4 NB03 pins before count
- `notebooks/05_train_direct_s2tt.ipynb` — reload semantic + file SHA; persist returns locked contract
- `tests/test_direct_s2tt.py` — round-trip / target / order / byte-tamper; 0/4–4/4 pin tests
- `configs/direct.yaml` — pins remain `""` (fail closed)

## New / updated tests

- Float/NaN/duration CSV round-trip: semantic PASS + byte SHA matches staged file
- Target change FAIL; order change FAIL; byte tamper FAIL
- Persist overwrites placeholder monitor SHA with staged-file SHA
- NB03 production 0/4 and 1–3/4 missing pins FAIL; 4/4 correct PASS; one wrong hash FAIL

## Verify results

- `python -m compileall -q src tests` — PASS
- Notebook 05 code cells AST-compile — 12 cells PASS
- `pytest -q` — **832 passed**
- No Hugging Face model download
- No notebook execute
- No Direct/ASR/MT pipeline stage run

## Base-source fingerprint

- Notebook 05 aggregate SHA-256: `72ccf303bad510867169186faf58c4db6da9c4f707bb14e33a6d9fa1ac28e8f5` (22 files)

## RunPod note

Before `FULL_STAGE="prepare"`, pin all four NB03 eligible hashes in `configs/direct.yaml`. Empty pins fail closed and print the computed hashes from the current NB03 CSVs — paste those four values, then prepare. Changing YAML after this zip updates the source fingerprint; that is intended. Recompute `SOURCE_FP` / re-prepare after the YAML pin.

Set `BAHNAR_DURABLE_CHECKPOINT_BUDGET_BYTES` before `resume_test_*` / `train`.

## Confirmation

No pipeline/model stage was run as part of this implementation.
