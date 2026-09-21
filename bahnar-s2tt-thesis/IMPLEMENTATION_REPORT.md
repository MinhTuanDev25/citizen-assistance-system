# Notebook 05 Direct S2TT — Implementation Report

Date: 2026-09-21 (follow-up: resume-safe Seq2SeqTrainer + no notebook budget default)

Phase B `data_position_ok` failed on RunPod because pinned `accelerate==1.10.1` `skip_first_batches()` rebuilds `DataLoaderShard` with `iteration=0`, so `SeedableRandomSampler` replays epoch-0 after a mid-epoch resume (global_step 20 → epoch 1, index 32). Model/optimizer/scheduler/RNG restore still passed. Cell 1 must not hard-code `BAHNAR_DURABLE_CHECKPOINT_BUDGET_BYTES`. **No prepare/pilot/resume/train was re-run in this edit.**

**READY_FOR_RUNPOD_PREPARE = True**. After this patch, restart the kernel and re-run `prepare` then continue the pipeline (source fingerprint changed).

## Spec gates closed

- Central `make_resume_safe_seq2seq_trainer_cls()` preserves dataloader `iteration` across `skip_first_batches` (upstream semantics: `epoch_dataloader.iteration = epochs_trained`).
- Used by `resume_test_b` (under UID tracking) and `FULL_STAGE=train` (`DirectTrainer`).
- Fresh pilot / Phase A keep stock `Seq2SeqTrainer`; fresh full train uses the same resume-safe class (no behavior change without resume skip).
- Resume proof not weakened: expected UID remains plan epoch-1 index 32; `data_position_ok` retained; Phase A still 20 steps; no `ignore_data_skip`.
- Cell 1 prints budget env if set; does not assign a notebook default.
- mBART `accepts_loss_kwargs=False` + embedding copy, contracts, monitor locks, peak copies, runtime pins unchanged.

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

- `src/direct_full_train.py` — `skip_first_batches_preserving_epoch` + `make_resume_safe_seq2seq_trainer_cls`
- `notebooks/05_train_direct_s2tt.ipynb` — wire resume-safe Trainer; remove Cell 1 budget hard-code
- `tests/test_direct_s2tt.py` — skip_first_batches iteration regression + Trainer resume UID integration
- `IMPLEMENTATION_REPORT.md` — fingerprint regenerated

## New / updated tests

- `skip_first_batches` alone yields epoch-0 UID at index 32; preserving iteration yields epoch-1 UID
- Continuous train past step 20 UID == resume-from-checkpoint-20 first UID; `data_position_ok`; step/optimizer/scheduler/RNG proofs
- Notebook AST requires `make_resume_safe_seq2seq_trainer_cls` wiring and forbids budget hard-code

## Verify results

- `python -m compileall -q src tests` — PASS
- `pytest -q tests/test_direct_s2tt.py` — **79 passed**
- No Hugging Face model download
- No notebook execute
- No Direct/ASR/MT pipeline stage run

## Base-source fingerprint

- Notebook 05 source fingerprint: `0926fa2f58b91ac021eab3f1097aa4b9d7ba1fd30aa7d725ece11069c5bf4de5` (22 files)

## RunPod note

Copy updated `src/direct_full_train.py` + notebook (+ tests/report if reviewing). Restart kernel, re-run `FULL_STAGE="prepare"` (fingerprint changed), then continue `pilot` → `resume_test_*` → `train`. Set `BAHNAR_DURABLE_CHECKPOINT_BUDGET_BYTES` in the environment (no notebook default).

## Confirmation

No pipeline/model stage was run as part of this implementation.
