# Notebook 06 port note

Date: 2026-09-22

Run-ready RQ1 evaluation hardening in the live thesis tree. Live NB03/NB04/NB05 sources were **not** overwritten.

## Live tree artifacts

- `notebooks/06_rq1_evaluation.ipynb`
- `configs/rq1.yaml`
- `src/rq1_*.py`
- `tests/test_rq1_evaluation.py`
- `NOTEBOOK06_IMPLEMENTATION_NOTES.md`
- `NOTEBOOK06_IMPLEMENTATION_REPORT.md`

## Hardening applied

- Legacy split_summary parquet pin via locked `rq1.yaml` + provenance source field
- Paired cluster bootstrap by `group_id` (bound into final contract)
- Full locked manifest bundle rehash after unlock
- Sequential C0 ASR→release→MT
- Robust project-root resolution + atomic durable writes
- Post-unlock identity + inference checkpoint digest binds (prior)

## Operator gate

Fill exact `ASR_STATE_DIR` / `MT_STATE_DIR` / `DIRECT_STATE_DIR` after NB03/04/05 expose durable best + `SUCCESS_*_EVALUATE`, then run verify → unlock_test → run_cascaded → run_direct → finalize.
