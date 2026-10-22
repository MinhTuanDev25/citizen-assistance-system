from __future__ import annotations

import importlib.metadata as metadata
import os
from pathlib import Path

import pytest

from inspect_runpod_lib import (
    ASR_REL,
    ASR_REQUIRED_FILES,
    LOCKED_TOKENIZER_ID,
    LOCKED_TOKENIZER_REVISION,
    MT_CONTRACT_REL,
    MT_REL,
    MT_REQUIRED_FILES,
    bounded_discover_audio,
    build_report,
    cas_root_from_script_file,
    checkpoint_presence,
    distribution_status,
    resolve_artifact_root,
    resolve_hf_hub_cache,
    resolve_layout,
    tokenizer_cache_status,
)


def test_cas_root_from_script_file():
    fake = Path("/tmp/cas root with spaces/scripts/phase7/inspect_runpod_phase7.sh")
    assert cas_root_from_script_file(fake) == Path("/tmp/cas root with spaces").resolve()


def test_resolve_layout_ok(tmp_path: Path):
    thesis = tmp_path / "bahnar-s2tt-thesis" / "src"
    thesis.mkdir(parents=True)
    (thesis / "c0_cascaded.py").write_text("# stub\n", encoding="utf-8")
    layout = resolve_layout(tmp_path)
    assert layout["layout_ok"] is True


def test_resolve_layout_mismatch(tmp_path: Path):
    layout = resolve_layout(tmp_path)
    assert layout["layout_ok"] is False
    assert "expected thesis entrypoint" in layout["mismatch"]


def test_audio_small_directory(tmp_path: Path):
    (tmp_path / "a.wav").write_bytes(b"RIFF")
    (tmp_path / "b.flac").write_bytes(b"fLaC")
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    out = bounded_discover_audio([tmp_path], max_depth=2, max_visits=50, max_results=8)
    names = {Path(p).name for p in out["candidates"]}
    assert names == {"a.wav", "b.flac"}
    assert out["truncated"] is False


def test_audio_empty_directory(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    out = bounded_discover_audio([empty])
    assert out["candidates"] == []
    assert out["count"] == 0


def test_audio_missing_directory(tmp_path: Path):
    missing = tmp_path / "nope"
    out = bounded_discover_audio([missing])
    assert out["candidates"] == []
    assert any(item.startswith("missing:") for item in out["truncation_reasons"])


def test_audio_large_tree_is_visit_bounded(tmp_path: Path):
    root = tmp_path / "tree"
    for i in range(40):
        d = root / f"d{i}"
        d.mkdir(parents=True)
        (d / f"{i}.wav").write_bytes(b"RIFF")
        (d / f"{i}.bin").write_bytes(b"x" * 8)
    out = bounded_discover_audio([root], max_depth=3, max_visits=15, max_results=50)
    assert out["truncated"] is True
    assert "visit_limit" in out["truncation_reasons"]
    assert out["visits"] <= 16
    assert out["count"] < 40


def test_audio_symlink_loop_does_not_hang(tmp_path: Path):
    a = tmp_path / "a"
    a.mkdir()
    (a / "hit.wav").write_bytes(b"RIFF")
    os.symlink(a, a / "loop")
    out = bounded_discover_audio([tmp_path], max_depth=8, max_visits=50, max_results=8)
    assert Path(out["candidates"][0]).name == "hit.wav"
    assert out["skipped_symlinks"] >= 1


def test_audio_permission_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    (blocked / "x.wav").write_bytes(b"RIFF")

    real_scandir = os.scandir

    def fake_scandir(path):
        if Path(path) == blocked:
            raise PermissionError("denied")
        return real_scandir(path)

    monkeypatch.setattr("inspect_runpod_lib.os.scandir", fake_scandir)
    out = bounded_discover_audio([tmp_path], max_depth=3, max_visits=50, max_results=8)
    assert out["permission_errors"] >= 1
    assert "x.wav" not in {Path(p).name for p in out["candidates"]}


def test_audio_result_limit(tmp_path: Path):
    for i in range(10):
        (tmp_path / f"{i}.wav").write_bytes(b"RIFF")
    out = bounded_discover_audio([tmp_path], max_depth=2, max_visits=100, max_results=3)
    assert out["count"] == 3
    assert out["truncated"] is True
    assert "result_limit" in out["truncation_reasons"]


def test_pyyaml_distribution_name():
    row = distribution_status("PyYAML", "yaml")
    assert row["distribution"] == "PyYAML"
    wrong = distribution_status("yaml", "yaml")
    assert wrong["distribution"] == "yaml"
    if wrong["distribution_error"] == "MISSING":
        assert wrong["status"] in {"MISSING", "NOT_VERIFIED"}


def test_missing_optional_package_does_not_raise():
    row = distribution_status("DefinitelyNotARealDistributionNameXYZ", "definitely_not_a_module_xyz")
    assert row["status"] in {"MISSING", "NOT_VERIFIED"}
    assert row["installed_version"] is None


def test_mt_required_files_match_source():
    src = Path(__file__).resolve().parents[2] / "bahnar-s2tt-thesis" / "src" / "c0_cascaded.py"
    text = src.read_text(encoding="utf-8")
    for name in MT_REQUIRED_FILES:
        assert f'"{name}"' in text
    assert MT_REQUIRED_FILES == (
        "model.safetensors",
        "config.json",
        "tokenizer_config.json",
        "sentencepiece.bpe.model",
        "dict.txt",
        "generation_config.json",
    )
    assert len(ASR_REQUIRED_FILES) == 5


def test_tokenizer_pin_matches_source():
    src = Path(__file__).resolve().parents[2] / "bahnar-s2tt-thesis" / "src" / "mt_contract.py"
    text = src.read_text(encoding="utf-8")
    assert LOCKED_TOKENIZER_ID in text
    assert LOCKED_TOKENIZER_REVISION in text
    assert LOCKED_TOKENIZER_REVISION == "36eee8b4d648dd99da56462edcda3c5c97f7f3de"


def _write_checkpoint(path: Path, names: tuple[str, ...]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for name in names:
        (path / name).write_text("stub\n", encoding="utf-8")


def _stub_cas(tmp_path: Path) -> Path:
    cas = tmp_path / "cas-source"
    src = cas / "bahnar-s2tt-thesis" / "src"
    src.mkdir(parents=True)
    (src / "c0_cascaded.py").write_text("# stub entrypoint\n", encoding="utf-8")
    cfg = cas / "bahnar-s2tt-thesis" / "configs"
    cfg.mkdir(parents=True)
    (cfg / "c0_inference.example.yaml").write_text("asr_checkpoint: \"\"\n", encoding="utf-8")
    return cas


def test_artifact_root_unset_does_not_use_thesis_source(tmp_path: Path):
    cas = _stub_cas(tmp_path)
    planted = cas / "bahnar-s2tt-thesis" / ASR_REL
    _write_checkpoint(planted, ASR_REQUIRED_FILES)
    resolved = resolve_artifact_root(explicit=None, env={}, documented=str(tmp_path / "missing-documented"))
    assert resolved["path"] is None
    assert resolved["source"] == "unset"
    report = build_report(
        cas,
        "python3",
        artifact_root=None,
        env={},
        documented_artifact_root=str(tmp_path / "missing-documented"),
    )
    assert report["artifact_root"]["source"] == "unset"
    assert report["asr_checkpoint"]["checkpoint_contract_verified"] == "FAIL"
    assert report["asr_checkpoint"]["path"] is None
    assert Path(cas / "bahnar-s2tt-thesis").resolve() != Path("/workspace/bahnar-s2tt-thesis")


def test_separate_cas_source_and_artifact_roots(tmp_path: Path):
    cas = _stub_cas(tmp_path)
    artifacts = tmp_path / "trained-artifacts"
    _write_checkpoint(artifacts / ASR_REL, ASR_REQUIRED_FILES)
    _write_checkpoint(artifacts / MT_REL, MT_REQUIRED_FILES)
    (artifacts / MT_CONTRACT_REL).parent.mkdir(parents=True, exist_ok=True)
    (artifacts / MT_CONTRACT_REL).write_text("{}\n", encoding="utf-8")
    (artifacts / "bahnar_s2tt" / "direct_full_state").mkdir(parents=True)
    planted_in_cas = cas / "bahnar-s2tt-thesis" / ASR_REL
    _write_checkpoint(planted_in_cas, ASR_REQUIRED_FILES)
    report = build_report(cas, "python3", artifact_root=artifacts, env={})
    assert Path(report["cas_root"]) == cas.resolve()
    assert Path(report["thesis_source_root"]) == (cas / "bahnar-s2tt-thesis").resolve()
    assert Path(report["artifact_root"]["path"]) == artifacts.resolve()
    assert report["asr_checkpoint"]["checkpoint_contract_verified"] == "PASS"
    assert report["mt_checkpoint"]["checkpoint_contract_verified"] == "PASS"
    assert report["mt_training_contract"]["files_found"] == "PASS"
    assert str(artifacts) in report["asr_checkpoint"]["path"]
    assert str(cas / "bahnar-s2tt-thesis") not in report["asr_checkpoint"]["path"]
    assert report["states"]["CHECKPOINT_CONTRACT_VERIFIED"] == "PASS"
    assert report["states"]["MODEL_LOAD_VERIFIED"] == "NOT_VERIFIED"


def test_incomplete_mt_checkpoint_still_fails(tmp_path: Path):
    cas = _stub_cas(tmp_path)
    artifacts = tmp_path / "trained-artifacts"
    _write_checkpoint(artifacts / ASR_REL, ASR_REQUIRED_FILES)
    _write_checkpoint(artifacts / MT_REL, MT_REQUIRED_FILES[:-1])
    report = build_report(cas, "python3", artifact_root=artifacts, env={})
    assert report["mt_checkpoint"]["checkpoint_contract_verified"] == "FAIL"
    assert "generation_config.json" in report["mt_checkpoint"]["missing"]
    assert report["states"]["CHECKPOINT_CONTRACT_VERIFIED"] == "FAIL"


def test_phase7_artifact_root_env(tmp_path: Path):
    art = tmp_path / "ext-root"
    art.mkdir()
    out = resolve_artifact_root(explicit=None, env={"PHASE7_ARTIFACT_ROOT": str(art)})
    assert out["source"] == "PHASE7_ARTIFACT_ROOT"
    assert Path(out["path"]) == art.resolve()
    assert out["exists"] is True


def test_documented_runpod_artifact_root_when_present(tmp_path: Path):
    documented = tmp_path / "workspace-bahnar-s2tt-thesis"
    (documented / "bahnar_s2tt").mkdir(parents=True)
    out = resolve_artifact_root(explicit=None, env={}, documented=str(documented))
    assert out["source"] == "documented_runpod_default"
    assert out["exists"] is True
    assert out["path"] == str(documented)


def test_cli_artifact_root_wins_over_env(tmp_path: Path):
    cli = tmp_path / "cli-root"
    env_root = tmp_path / "env-root"
    cli.mkdir()
    env_root.mkdir()
    out = resolve_artifact_root(
        explicit=cli,
        env={"PHASE7_ARTIFACT_ROOT": str(env_root)},
        documented=str(tmp_path / "documented"),
    )
    assert out["source"] == "cli"
    assert Path(out["path"]) == cli.resolve()


def test_checkpoint_presence_gates_all_required_files(tmp_path: Path):
    asr = tmp_path / "asr"
    _write_checkpoint(asr, ASR_REQUIRED_FILES)
    assert checkpoint_presence(asr, ASR_REQUIRED_FILES)["checkpoint_contract_verified"] == "PASS"
    mt = tmp_path / "mt"
    _write_checkpoint(mt, MT_REQUIRED_FILES)
    assert checkpoint_presence(mt, MT_REQUIRED_FILES)["checkpoint_contract_verified"] == "PASS"
    (mt / "dict.txt").unlink()
    failed = checkpoint_presence(mt, MT_REQUIRED_FILES)
    assert failed["checkpoint_contract_verified"] == "FAIL"
    assert "dict.txt" in failed["missing"]


def test_hf_hub_cache_env_does_not_append_hub():
    out = resolve_hf_hub_cache({"HF_HUB_CACHE": "/custom/hf-hub-cache"})
    assert out["cache_root"] == "/custom/hf-hub-cache"
    assert out["source"] == "HF_HUB_CACHE"
    assert out["appended_hub"] is False
    assert not out["cache_root"].endswith("/hub")


def test_hf_hub_cache_wins_over_legacy_and_hf_home():
    out = resolve_hf_hub_cache(
        {
            "HF_HUB_CACHE": "/new/hub-cache",
            "HUGGINGFACE_HUB_CACHE": "/legacy/hub-cache",
            "HF_HOME": "/data/hf",
        }
    )
    assert out["cache_root"] == "/new/hub-cache"
    assert out["source"] == "HF_HUB_CACHE"
    assert out["appended_hub"] is False


def test_hf_hub_cache_does_not_append_hub_when_override_is_hub_dir():
    out = resolve_hf_hub_cache({"HUGGINGFACE_HUB_CACHE": "/data/hf/hub"})
    assert out["cache_root"] == "/data/hf/hub"
    assert out["source"] == "HUGGINGFACE_HUB_CACHE"
    assert out["appended_hub"] is False


def test_hf_hub_cache_does_not_double_hub():
    out = resolve_hf_hub_cache({"HUGGINGFACE_HUB_CACHE": "/custom/already-hub"})
    assert out["cache_root"] == "/custom/already-hub"
    assert not out["cache_root"].endswith("/hub/hub")


def test_hf_home_appends_hub_once():
    out = resolve_hf_hub_cache({"HF_HOME": "/data/hf"})
    assert out["cache_root"] == "/data/hf/hub"
    assert out["source"] == "HF_HOME/hub"
    assert out["appended_hub"] is True


def test_huggingface_hub_cache_wins_over_hf_home():
    out = resolve_hf_hub_cache(
        {
            "HUGGINGFACE_HUB_CACHE": "/custom/hub-cache",
            "HF_HOME": "/data/hf",
        }
    )
    assert out["cache_root"] == "/custom/hub-cache"
    assert out["source"] == "HUGGINGFACE_HUB_CACHE"


def test_tokenizer_cache_distinguishes_dir_snapshot_and_load(tmp_path: Path):
    hub = tmp_path / "hub"
    hub.mkdir()
    missing_snap = tokenizer_cache_status({"HUGGINGFACE_HUB_CACHE": str(hub)})
    assert missing_snap["cache_dir_exists"] == "PASS"
    assert missing_snap["snapshot_revision_present"] == "MISSING"
    assert missing_snap["tokenizer_contract_validated"] == "NOT_VERIFIED"
    assert missing_snap["tokenizer_loaded"] == "NOT_VERIFIED"
    assert missing_snap["tokenizer_verified"] == "NOT_VERIFIED"
    snap = hub / "models--vinai--bartpho-syllable" / "snapshots" / LOCKED_TOKENIZER_REVISION
    snap.mkdir(parents=True)
    present = tokenizer_cache_status({"HUGGINGFACE_HUB_CACHE": str(hub)})
    assert present["snapshot_revision_present"] == "PASS"
    assert present["files_found"] == "PASS"
    assert present["tokenizer_loaded"] == "NOT_VERIFIED"
    assert present["tokenizer_contract_validated"] == "NOT_VERIFIED"


def test_tokenizer_pin_not_disabled():
    assert LOCKED_TOKENIZER_ID == "vinai/bartpho-syllable"
    assert LOCKED_TOKENIZER_REVISION == "36eee8b4d648dd99da56462edcda3c5c97f7f3de"


def test_pyyaml_real_distribution_if_installed():
    try:
        ver = metadata.version("PyYAML")
    except metadata.PackageNotFoundError:
        pytest.skip("PyYAML not installed in this interpreter")
    row = distribution_status("PyYAML", "yaml")
    assert row["installed_version"] == ver
    assert row["imported"] is True
    assert row["status"] == "PASS"
