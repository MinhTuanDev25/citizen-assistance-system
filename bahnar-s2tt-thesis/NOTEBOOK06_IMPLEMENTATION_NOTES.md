# Notebook 06 implementation notes

## Operator controls

Only the notebook cell tagged `rq1-operator-controls` is intended to change between stages:

- `RQ1_STAGE`
- `ALLOW_FROZEN_TEST_ACCESS`
- `ASR_STATE_DIR`
- `MT_STATE_DIR`
- `DIRECT_STATE_DIR`
- `DEVICE`

That cell is deliberately excluded from the scientific source fingerprint. All other NB06 scientific/orchestration cells are fingerprinted.

## Run requirements

- `verify`: no GPU required, but checkpoint durable storage and enough temporary local disk for one restored experiment tree at a time are required.
- `unlock_test`: no GPU required; first real frozen-test access.
- `run_cascaded`: CUDA required.
- `run_direct`: CUDA required.
- `finalize`: GPU not required.

## Methodology boundary

After `unlock_test` succeeds, do not use frozen-test results to retune or retrain NB03, NB04, or NB05. If an upstream scientific artifact changes, the handoff/final contract identity changes and old downstream artifacts are rejected.
