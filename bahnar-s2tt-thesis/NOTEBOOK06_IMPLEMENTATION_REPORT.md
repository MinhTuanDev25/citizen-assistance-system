# Notebook 06 RQ1 Evaluation — Implementation Report

Date: 2026-09-22 (final runtime-hardening: exception-safe cleanup + fail-closed bootstrap)

Hardening for **real** end-to-end RQ1 execution readiness. **No notebook execute, no real rq1_test content open, no model/dataset download, no inference, no training.** Checkpoint load proof / cascade sequencing / failure-path tests use **injectable mocks only**.

NB05 `IMPLEMENTATION_REPORT.md` is left unchanged (Direct fingerprint gate).

## Root causes fixed (this pass)

1. **Exception-safe local restore cleanup** — `run_cascaded_c0_peak_safe` / `run_direct_d0_peak_safe` use `try/finally` so proof, inference, UID validation, and write failures still release CUDA and delete local trees. `restore_*_for_rq1` also cleans on load failure after durable materialization.
2. **Strict `allowed_root` deletion** — `cleanup_local_rq1_experiment(path, allowed_root=…)` requires a resolved strict descendant; refuses roots (`/`, `/tmp`, allowed_root itself); never silently ignores cleanup failures; verify-stage uses the same helper.
3. **ASR UID before MT restore** — exact locked UID order validated after ASR inference and before MT restore; again after MT for C0.
4. **`bootstrap_ok` required** — `derive_final_status` has no default; `SUCCESS_RQ1_FINAL` cannot omit the bootstrap gate.
5. **C0/D0 summary bind** — finalize requires `rq1_final_contract_hash` + `n == test_count` via `assert_stage_summary_matches_final_contract`.

## Prior V4 scientific gates retained

Verify cannot full-read frozen test; `unlock_test` first frozen read; `ALLOW_FROZEN_TEST_ACCESS`; exact manifest SHA pins; legacy locked parquet provenance; exact 215 test rows; overlap protection; post-unlock identity; checkpoint content digest; same decoded PCM C0/D0; exact ordered UID; upstream generation settings; cluster bootstrap by `group_id`; bootstrap finite CI gate; atomic writes; source/runtime fingerprint.

## Files changed

- `src/rq1_restore.py` — strict cleanup + peak-safe try/finally + restore failure cleanup
- `src/rq1_contract.py` — required `bootstrap_ok`; UID/summary helpers
- `notebooks/06_rq1_evaluation.ipynb` — `allowed_root` / `expected_uids` / summary bind / bootstrap gate
- `tests/test_rq1_evaluation.py` — failure-path + path-safety + stale-summary regressions
- `NOTEBOOK06_IMPLEMENTATION_REPORT.md` — this report

## New failure-path / safety tests

- `test_cleanup_allowed_root_safety`
- `test_failure_paths_cleanup_before_next_restore`
- `test_asr_uid_validated_before_mt_restore`
- `test_restore_load_failure_cleans_local_tree`
- `test_stage_summary_must_match_current_final_contract`
- `test_bootstrap_gate_required_for_final_success`
- `test_notebook_wires_disk_cleanup_and_bootstrap_gate`

## Verify results

- `python -m compileall -q src tests` — PASS
- Notebook 06 code cells AST-compile — **9 cells** PASS
- `pytest -q tests/` — **868 passed**, 0 failed
- No Hugging Face model/dataset download
- No notebook execute / no RQ1 pipeline stage run

## Base-source fingerprint

- Notebook 06 source fingerprint: `c3718fadce6a09984b75f4b8c4ea032c7b8ea4e744ff470cfd07c94343447c85` (29 files)

## Confirmation

No frozen-test / inference / training stage was run as part of this implementation.

## Readiness

Code is ready for the operator sequence:

`verify` → `unlock_test` → `run_cascaded` → `run_direct` → `finalize`
