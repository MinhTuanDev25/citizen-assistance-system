"""Synthetic fixtures for Notebook 13. Nothing here touches a real artifact tree."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from src.rq1_contract import sha256_file, sha256_json
from src.rq2_pseudo_contract import (
    PSEUDO_SCHEMA_VERSION,
    STATUS_SUCCESS,
    finalize_generation,
    stage_generation,
    write_json,
)
from src.rq2_quality import NB12_ARTIFACT_FILES, U_PRIME_COLUMNS
from src.rq2_selection_contract import ordered_uid_sha256
from tests.rq2_nb12_fixtures import build_nb11_generation


def u_prime_row(uid: str, duration: float, quality: float, text: str = "xin chào", **overrides: Any) -> Dict[str, Any]:
    row = {column: None for column in U_PRIME_COLUMNS}
    row.update({
        "segment_uid": uid,
        "source_id": "src-a",
        "source_group_id": "grp-a",
        "segment_local_path": f"artifacts/rq2/u_clean/segments/src-a/{uid}.wav",
        "segment_pcm16_sha256": (uid.encode("utf-8").hex() + "ab" * 32)[:64],
        "segment_wav_sha256": (uid.encode("utf-8").hex() + "cd" * 32)[:64],
        "duration_seconds": float(duration),
        "asr_text_norm": "bahnar",
        "pseudo_vi_norm": text,
        "asr_mean_logprob": -0.2,
        "mt_mean_logprob": -0.4,
        "asr_confidence_norm": 0.5,
        "mt_confidence_norm": 0.6,
        "quality_penalty": 0.0,
        "quality_score": float(quality),
        "nb11_generation_id": "gen",
        "nb11_input_contract_sha256": "11" * 32,
        "teacher_contract_sha256": "22" * 32,
        "d0_agreement_contract_sha256": "33" * 32,
        "quality_score_contract_sha256": "44" * 32,
    })
    row.update(overrides)
    return {column: row.get(column) for column in U_PRIME_COLUMNS}


def selection_identity(rows: Sequence[Dict[str, Any]], **overrides: Any) -> Dict[str, Any]:
    identity = {
        "nb11_generation_id": "nb11-gen",
        "nb11_input_contract_sha256": "11" * 32,
        "nb12_generation_id": "nb12-gen",
        "nb12_contract_sha256": "55" * 32,
        "u_prime_manifest_sha256": "66" * 32,
        "u_prime_ordered_uid_sha256": ordered_uid_sha256([row["segment_uid"] for row in rows]),
        "quality_score_contract_sha256": "44" * 32,
    }
    identity.update(overrides)
    return identity


def write_u_prime_parquet(path: Path, rows: Sequence[Dict[str, Any]]) -> str:
    ordered = [{column: row.get(column) for column in U_PRIME_COLUMNS} for row in rows]
    table = pa.Table.from_pylist(ordered)
    if list(table.column_names) != list(U_PRIME_COLUMNS):
        table = table.select(list(U_PRIME_COLUMNS))
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return sha256_file(path)


def seal_nb12_generation(project_root: Path, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Publish a hash-sealed NB12 generation bound to a synthetic frozen NB11 input."""
    from src.rq2_pseudo_contract import resolve_nb11_input

    build_nb11_generation(project_root, n_segments=max(4, len(rows)))
    nb11 = resolve_nb11_input(project_root)
    bound = []
    for index, row in enumerate(rows):
        source = nb11.rows[index]
        updated = dict(row)
        updated.update({
            "segment_uid": source["segment_uid"],
            "source_id": source["source_id"],
            "source_group_id": source["source_group_id"],
            "segment_local_path": source["segment_local_path"],
            "segment_pcm16_sha256": source["segment_pcm16_sha256"],
            "segment_wav_sha256": source["segment_wav_sha256"],
            "duration_seconds": float(source["duration_seconds"]),
            "nb11_generation_id": nb11.generation_id,
            "nb11_input_contract_sha256": nb11.contract_sha256,
        })
        bound.append({column: updated.get(column) for column in U_PRIME_COLUMNS})
    pseudo = project_root / "artifacts" / "rq2" / "pseudo_labels"
    staged = stage_generation(pseudo, gen_id="nb12-fixture-gen")
    gen_dir = staged["staging_dir"]
    write_u_prime_parquet(gen_dir / "u_prime_manifest.parquet", bound)
    shutil.copyfile(gen_dir / "u_prime_manifest.parquet", gen_dir / "pseudo_label_manifest.parquet")
    write_json(gen_dir / "nb11_input_contract.json", nb11.contract)
    contract = {
        "schema_version": PSEUDO_SCHEMA_VERSION,
        "nb11_input_contract_sha256": nb11.contract_sha256,
        "nb11_generation_id": nb11.generation_id,
        "teacher_contract_sha256": "22" * 32,
        "d0_agreement_contract_sha256": "33" * 32,
        "decoding_config_sha256": "77" * 32,
        "validation_input_contract_sha256": "88" * 32,
        "validation_audio_identity_sha256": "99" * 32,
        "calibration_artifact_sha256": "aa" * 32,
        "quality_score_contract_sha256": "44" * 32,
        "quality_score_proposal_sha256": "bb" * 32,
        "u_prime_is_common_pool_for_all_rq2_treatments": True,
    }
    contract["nb12_contract_sha256"] = sha256_json(contract)
    write_json(gen_dir / "contract.json", contract)
    summary = {
        "status": STATUS_SUCCESS,
        "schema_version": PSEUDO_SCHEMA_VERSION,
        "quality_score_frozen": True,
        "nb12_contract_sha256": contract["nb12_contract_sha256"],
        "u_prime_manifest_sha256": sha256_file(gen_dir / "u_prime_manifest.parquet"),
        "generation_id": staged["generation_id"],
    }
    write_json(gen_dir / "summary.json", summary)
    for name in NB12_ARTIFACT_FILES:
        target = gen_dir / name
        if target.exists():
            continue
        if name.endswith(".jsonl"):
            target.write_text("{}\n", encoding="utf-8")
        else:
            write_json(target, {"fixture": True})
    finalize_generation(pseudo, staged, NB12_ARTIFACT_FILES)
    return {"rows": bound, "contract": contract, "nb11": nb11, "pseudo_dir": pseudo}
