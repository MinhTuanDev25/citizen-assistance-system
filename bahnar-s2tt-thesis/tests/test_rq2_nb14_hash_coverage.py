"""Scientific artifact hash coverage for dynamic NB14 generation files."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from src.rq2_final_contract import STATUS_SUCCESS
from src.rq2_final_evaluate import (
    SCIENTIFIC_HASH_EXCLUDED_BASENAMES,
    STATUS_FAIL,
    hash_artifact_files,
    iter_scientific_artifact_files,
    publish_final_generation,
)
from src.rq2_pseudo_contract import write_json
from tests.test_rq2_nb14_orchestration import (
    _assert_hash_coverage,
    _dynamic_hash_paths,
    _orchestrate,
    _publish,
)


def test_enumeration_excludes_manifest_lifecycle_and_temp_files(tmp_path):
    (tmp_path / "predictions").mkdir()
    (tmp_path / "predictions" / "d0.parquet").write_bytes(b"core")
    (tmp_path / "evaluation" / "seed-3").mkdir(parents=True)
    (tmp_path / "evaluation" / "seed-3" / "bootstrap.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "artifact_hashes.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "COMPLETE.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "artifact_manifest.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "CURRENT").write_text("pointer\n", encoding="utf-8")
    (tmp_path / "scratch.tmp").write_text("temp\n", encoding="utf-8")
    (tmp_path / "staging.partial").mkdir()
    (tmp_path / "staging.partial" / "leak.json").write_text("{}\n", encoding="utf-8")
    found = iter_scientific_artifact_files(tmp_path)
    assert found == [
        "evaluation/seed-3/bootstrap.json",
        "predictions/d0.parquet",
    ]
    assert SCIENTIFIC_HASH_EXCLUDED_BASENAMES == frozenset({
        "artifact_hashes.json",
        "COMPLETE.json",
        "artifact_manifest.json",
        "CURRENT",
    })
    hashed = hash_artifact_files(tmp_path)
    assert list(hashed) == found


def test_dynamic_hash_tamper_coverage_and_current_atomicity(tmp_path):
    built = _orchestrate(tmp_path / "source", [7, 11, 13])
    _result, mapping, artifact_dir = _publish(tmp_path / "source", built)
    assert _result["status"] == STATUS_SUCCESS, _result.get("error")
    current = (built["env"]["durable_root"] / "artifacts" / "rq2" / "final" / "CURRENT").read_text().strip()
    generation = built["env"]["durable_root"] / "artifacts" / "rq2" / "final" / "generations" / current
    _assert_hash_coverage(generation, [7, 11, 13])
    for rel in _dynamic_hash_paths([7, 11, 13]):
        assert rel in json.loads((artifact_dir / "artifact_hashes.json").read_text())["files"]

    pointer = built["env"]["durable_root"] / "artifacts" / "rq2" / "final" / "CURRENT"
    sealed = pointer.read_text(encoding="utf-8")

    def clone(name):
        dest = tmp_path / name
        shutil.copytree(artifact_dir, dest)
        cloned = {rel: dest / rel for rel in mapping}
        return dest, cloned

    def fail(name, mutate):
        dest, cloned = clone(name)
        mutate(dest, cloned)
        result = publish_final_generation(
            tmp_path / name,
            artifacts=cloned,
            durable_root=str(built["env"]["durable_root"]),
        )
        assert result["status"] == STATUS_FAIL, result
        assert result["wrote_current"] is False
        assert pointer.read_text(encoding="utf-8") == sealed
        return result

    def touch(path: Path):
        path.write_bytes(path.read_bytes() + b"\n")

    quality = fail("quality", lambda dest, _cloned: touch(dest / "evaluation/seed-11/d_quality.parquet"))
    assert "artifact hash mismatch: evaluation/seed-11/d_quality.parquet" in quality["error"]
    bootstrap = fail("bootstrap", lambda dest, _cloned: touch(dest / "evaluation/seed-11/bootstrap.json"))
    assert "artifact hash mismatch: evaluation/seed-11/bootstrap.json" in bootstrap["error"]
    aggregate = fail("aggregate", lambda dest, _cloned: touch(dest / "aggregate/metrics.json"))
    assert "artifact hash mismatch: aggregate/metrics.json" in aggregate["error"]
    training = fail("training", lambda dest, _cloned: touch(dest / "training/d_random/seed-11.json"))
    assert "artifact hash mismatch: training/d_random/seed-11.json" in training["error"]

    def drop_hash(dest, _cloned):
        path = dest / "artifact_hashes.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        del payload["files"]["evaluation/seed-11/bootstrap.json"]
        write_json(path, payload)

    dropped = fail("dropped", drop_hash)
    assert "missing hash coverage" in dropped["error"]
    assert "un-hashed scientific artifact" in dropped["error"]
    assert "evaluation/seed-11/bootstrap.json" in dropped["error"]

    def extra_file(dest, cloned):
        rel = "evaluation/seed-11/extra.json"
        path = dest / rel
        path.write_text('{"extra": true}\n', encoding="utf-8")
        cloned[rel] = path

    extra = fail("extra", extra_file)
    assert "missing hash coverage" in extra["error"]
    assert "un-hashed scientific artifact" in extra["error"]
    assert "evaluation/seed-11/extra.json" in extra["error"]

    def ghost(dest, _cloned):
        path = dest / "artifact_hashes.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["files"]["evaluation/seed-11/missing.parquet"] = "ab" * 32
        write_json(path, payload)

    missing = fail("ghost", ghost)
    assert "lists a missing artifact" in missing["error"]
    assert "evaluation/seed-11/missing.parquet" in missing["error"]
    assert pointer.read_text(encoding="utf-8") == sealed
