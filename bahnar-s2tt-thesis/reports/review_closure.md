# Step 7.2 review closure — provenance semantics (Option 1)

Date: 2026-10-15. Source semantics only. No GPU run. No Hub download.
No invented research SHA-256. No model registry.

This round closes the last HIGH finding: false `verified_c0` /
`research_provenance=VERIFIED` from operator-supplied research pins.
Checksum fail-closed and manifest CLI JSON from the previous round are
unchanged.

## HIGH — False research provenance verification

**Root cause.** `load_research_pins` (`src/c0_cascaded.py` 196–226) reads a
YAML path from serving config (`research_pins:` or
`configs/c0_research_pins.yaml`). `assess_checkpoint` then treated a SHA
match against that file as independent research attestation:

```json
{"checkpoint_identity": "verified_c0", "research_provenance": "VERIFIED"}
```

Independent review reproduced fake checkpoint + fake pins → `verified_c0`.
The unit test `test_research_pin_match_is_verified_c0` encoded the same
overclaim. Two operator YAML files are not a locked research source.

**Before (false VERIFIED).**

```json
{"checkpoint_identity": "verified_c0", "research_provenance": "VERIFIED"}
```

Assignment was `src/c0_cascaded.py` `assess_checkpoint` (old lines 295–297)
and pair identity (old 330–331). `C0Cascade` / `build_manifest` copied
`VERIFIED` to JSON. Experiment IDs `full_xlsr300m_v1` /
`mt_bartpho_syllable_v1` were emitted on that label.

**After (Option 1).**

```json
{
  "checkpoint_identity": "research_reference_hash_match",
  "research_provenance": "REFERENCE_HASH_MATCH"
}
```

- `assess_checkpoint` lines 301–302: research-pin match →
  `research_reference_hash_match` / `REFERENCE_HASH_MATCH`.
- `pair_checkpoint_identity` lines 335–336 and `pair_research_provenance`
  lines 346–350: pair-level labels; never `verified_c0` or `VERIFIED`.
- `C0Cascade` uses `pair_research_provenance` (line 673).
- `src/c0_manifest.py` `build_manifest` line 164 uses the same helper.
- Experiment IDs are not emitted. Manifest `experiment_id` is always `null`
  until an independent research source exists (it does not, in this repo).
- Source no longer contains `identity = "verified_c0"` or
  `research_provenance = "VERIFIED"`. Tokenizer `tokenizer_parity=VERIFIED`
  is a different field and is unchanged.

Operator checksum is unchanged: declared YAML SHA mismatch still raises
`checkpoint_checksum_mismatch` in `assert_operator_checksum` (lines 229–247)
before `load_asr_bundle` / `load_mt_bundle` (lines 642–643). Research-pin
mismatch does **not** fail closed; it stays `NOT_VERIFIED` /
not-`research_reference_hash_match`.

**Regression tests.**
- Fake checkpoint + fake pins → `REFERENCE_HASH_MATCH`, not `verified_c0`
- Structural + reference SHA match → same, not independent VERIFIED
- Missing research pins → `NOT_VERIFIED`
- Reference mismatch → not VERIFIED, not reference match
- Operator SHA mismatch → fail closed, loader not called
- Inference JSON, `build_manifest`, and manifest CLI agree on the same labels
- Source scan: no `identity = "verified_c0"` assignment

**Actual results.** `48 passed, 0 failed, 0 skipped` (pytest `--noconftest`).
`compileall` exit 0. Import smoke `import_ok`. Tiny fake weights only.

**Remaining limitations.**
- `REFERENCE_HASH_MATCH` still means “weights match the configured pointer”.
  An operator can still create that pointer. The label no longer claims
  independent research verification.
- `verified_c0` / `research_provenance=VERIFIED` are currently unreachable
  from C0 serving. That is intended until a locked research hash exists
  outside this module.
- `REAL_RESEARCH_PROVENANCE` remains `NOT_VERIFIED`.

## Preserved from prior rounds

Tokenizer Hub pin + fingerprint, empty ASR/MT serving gates, one-time
audio read, finite duration, generation identity guard, inference CLI
JSON, operator SHA fail-closed, manifest CLI JSON, ASR/MT decode path.

## Gates (this execution)

```text
SOURCE_IMPLEMENTATION=PASS
UNIT_TESTS=PASS (48 passed, mock/tiny-file only)
PROVENANCE_SEMANTICS=PASS
CHECKPOINT_INTEGRITY_CODE=PASS
SOURCE_READY=PASS

REAL_RESEARCH_PROVENANCE=NOT_VERIFIED
REAL_TOKENIZER_PARITY=NOT_VERIFIED
MODEL_LOAD=NOT_RUN
REAL_AUDIO_INFERENCE=NOT_RUN
STEP_7_3_READINESS=READY_TO_RUN_ON_RUNPOD
```

`SOURCE_READY=PASS` means the provenance label no longer overclaims.
It is not real-inference-ready.
