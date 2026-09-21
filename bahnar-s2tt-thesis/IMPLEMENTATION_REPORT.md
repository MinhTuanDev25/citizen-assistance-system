# Notebook 05 Direct S2TT — Implementation Report

Date: 2026-09-21 (follow-up: Trainer-only `num_items_in_batch` fix + mBART embedding warm-start)

Pilot crashed because transformers 4.57 `Seq2SeqTrainer` injects `num_items_in_batch` into `SpeechEncoderDecoder`, which forwards it to `MBartForCausalLM.forward()`. The same RunPod log showed `lm_head.weight` and `model.decoder.embed_tokens.weight` were newly initialized when the mmt checkpoint was loaded as CausalLM. **No prepare/pilot/train was re-run in this edit.**

**READY_FOR_RUNPOD_PREPARE = True** after NB03 eligible hashes are pinned. After this patch, restart the kernel and re-run `prepare` then `pilot` (source fingerprint changed).

## Spec gates closed

- Root fix for pilot: `model.accepts_loss_kwargs = False` only. No decoder.forward monkey-patch.
- Keep `copy_mbart_seq2seq_embeddings()`: load `MBartForConditionalGeneration` at the locked decoder id/revision and copy `shared` + `lm_head` into the CausalLM decoder (shape mismatch fails closed).
- Regression: construct a real `Seq2SeqTrainer` and assert `trainer.model_accepts_loss_kwargs is False` (no `train()`).
- Monitor semantic/byte locks, 4/4 NB03 pins, peak copies, runtime pins, data/training contracts unchanged.

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

- `src/direct_model.py` — `accepts_loss_kwargs=False`; copy mBART shared/`lm_head`; removed `drop_unexpected_decoder_loss_kwargs`
- `tests/test_direct_s2tt.py` — Trainer-level loss-kwargs regression; embedding copy + shape fail-closed
- `IMPLEMENTATION_REPORT.md` — fingerprint regenerated

## New / updated tests

- `Seq2SeqTrainer(...).model_accepts_loss_kwargs is False` after `load_direct_model`
- mBART shared + `lm_head` copy PASS; shape mismatch FAIL
- No decoder.forward monkey-patch path remains

## Verify results

- `python -m compileall -q src tests` — PASS
- `pytest -q tests/test_direct_s2tt.py` — **77 passed**
- `pytest -q` — **835 passed**
- No Hugging Face model download
- No notebook execute
- No Direct/ASR/MT pipeline stage run

## Base-source fingerprint

- Notebook 05 source fingerprint: `05d5f9b9a893c46a1f9aac10e138d4112bfdedb65a397c91ee6b3c1473eae327` (22 files)

## RunPod note

Copy updated `src/direct_model.py` (+ tests/report if reviewing). Restart kernel, re-run `FULL_STAGE="prepare"` (fingerprint changed), then `pilot`. Set `BAHNAR_DURABLE_CHECKPOINT_BUDGET_BYTES` before `resume_test_*` / `train`.

## Confirmation

No pipeline/model stage was run as part of this implementation.
