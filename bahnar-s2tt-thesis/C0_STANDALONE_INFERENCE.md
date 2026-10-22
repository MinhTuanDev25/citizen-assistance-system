# C0 cascaded standalone inference (Step 7.2)

Date: 2026-10-15. No GPU inference. No Hub tokenizer download on this machine.

Repo config directory is `configs/`, not `config/`.

## CLI

```bash
python -m src.c0_cascaded --config configs/c0_inference.example.yaml --audio /path/to/bahnar.wav
python -m src.c0_manifest --config configs/c0_inference.example.yaml --out manifests/c0_manifest.json
python -m src.c0_tokenizer --contract /path/to/mt_training_contract.json
python -m src.c0_speech   # internal HTTP API; see C0_SPEECH_API.md
```

`checkpoint_identity` (do not collapse these):

- `expected_step_match` — basename is `checkpoint-12500` / `checkpoint-17500`
- `configured_hash_match` — operator YAML SHA-256 matches `model.safetensors`
- `research_reference_hash_match` — weights match `configs/c0_research_pins.yaml`

`research_provenance`:

- `NOT_VERIFIED` — no research-reference pin, or the pin does not match
- `REFERENCE_HASH_MATCH` — weights match the configured research-reference file

Configured YAML files never mint `verified_c0` or `research_provenance=VERIFIED`.
Basename-only paths, including missing directories, are never C0-verified.
A declared operator SHA that does not match fails closed
(`checkpoint_checksum_mismatch`) before model load.

Successful inference is not research provenance.

`generation_matches_rq1` is true only for 256/256/4.

Manifest `--out` creates missing parent directories. Config and output
failures emit JSON (`config_error` / `output_error`) and exit 1.

## Tests

```text
python -m pytest -q tests/test_c0_cascaded.py tests/test_c0_tokenizer.py tests/test_c0_speech.py
```

## Status

```text
SOURCE_IMPLEMENTATION=PASS
UNIT_TESTS=see reports/test_results.txt
CHECKPOINT_INTEGRITY=LOGIC_PASS
PROVENANCE_SEMANTICS=PASS
RESEARCH_CHECKPOINT_PROVENANCE=NOT_VERIFIED
TOKENIZER_PARITY=NOT_VERIFIED
MODEL_LOAD=NOT_RUN
REAL_AUDIO_INFERENCE=NOT_RUN
STEP_7_3_READINESS=READY_TO_RUN_ON_RUNPOD
```
