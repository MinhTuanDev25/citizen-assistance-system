"""Unit tests for C0 standalone inference. Models are mocked; this is not GPU evidence."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.c0_cascaded import (
    ASR_PROCESSOR_KWARGS,
    ASR_REQUIRED_FILES,
    DEFAULT_MAX_AUDIO_DURATION_SECONDS,
    MODEL_ID,
    MT_REQUIRED_FILES,
    SAMPLE_RATE,
    TRAINING_STATE_FILES,
    C0AudioError,
    C0Cascade,
    C0ChecksumError,
    C0ConfigError,
    assess_checkpoint,
    assert_checkpoint_ready,
    checkpoint_identity,
    checkpoint_step,
    load_c0_config,
    main as c0_main,
    prepare_asr_waveforms,
    read_c0_audio,
    resolve_device,
)
from src.c0_manifest import build_manifest, main as manifest_main
from src.data_utils import normalize_bahnar_ctc_v1

ROOT = Path(__file__).resolve().parents[1]
# Research-allowed Bahnar strings already used in thesis tests / notebook 03 artifacts.
RESEARCH_BAHNAR = (
    "Pơlei Bahnar",
    "Inh jua",
    "Kơthơ inh chih ăn kơ iĕm ki, inh khă kuă ăn iĕm tơbôl păng bơngai pơm kư̆ kă.",
)


def _touch_checkpoint(root: Path, names: tuple[str, ...]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        (root / name).write_bytes(b"not-a-weight")


def _config(tmp_path: Path, *, verified_names: bool = False) -> dict:
    asr = tmp_path / ("checkpoint-12500" if verified_names else "asr")
    mt = tmp_path / ("checkpoint-17500" if verified_names else "mt")
    _touch_checkpoint(asr, ASR_REQUIRED_FILES)
    _touch_checkpoint(mt, MT_REQUIRED_FILES)
    return {
        "asr_checkpoint": str(asr),
        "mt_checkpoint": str(mt),
        "device": "cpu",
        "max_source_length": 256,
        "generation_max_length": 256,
        "num_beams": 4,
        "max_audio_duration_seconds": 30,
        "max_audio_bytes": 10485760,
        "research_pins": "",
    }


def _reader(wav: np.ndarray, sr: int):
    def _read(_path, dtype="float32", always_2d=False):
        del dtype, always_2d
        return wav, sr

    return _read


def _cascade(tmp_path: Path, transcribe, translate, wav: np.ndarray, *, verified_names: bool = False, extra: dict | None = None, **kwargs):
    cfg = _config(tmp_path, verified_names=verified_names)
    if extra:
        cfg.update(extra)
    return C0Cascade(
        cfg,
        asr_bundle={"model": "asr-model", "processor": "asr-processor", "checkpoint": str(tmp_path / ("checkpoint-12500" if verified_names else "asr"))},
        mt_bundle={
            "model": "mt-model",
            "tokenizer": "mt-tokenizer",
            "checkpoint": str(tmp_path / ("checkpoint-17500" if verified_names else "mt")),
            "tokenizer_parity": kwargs.get("tokenizer_parity", "NOT_VERIFIED"),
            "tokenizer_source": "hub_pinned_rq1",
            "tokenizer_model_id": "vinai/bartpho-syllable",
            "tokenizer_revision": "36eee8b4d648dd99da56462edcda3c5c97f7f3de",
        },
        transcribe=transcribe,
        translate=translate,
        audio_reader=_reader(wav, 16000),
    )


def test_audio_rejects_wrong_rate_empty_and_non_finite(tmp_path: Path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    with pytest.raises(C0AudioError, match="16000"):
        read_c0_audio(audio, reader=_reader(np.zeros(16, dtype=np.float32), 8000))
    with pytest.raises(C0AudioError, match="empty"):
        read_c0_audio(audio, reader=_reader(np.zeros(0, dtype=np.float32), 16000))
    with pytest.raises(C0AudioError, match="non-finite"):
        read_c0_audio(audio, reader=_reader(np.array([np.nan], dtype=np.float32), 16000))
    stereo = np.ones((8, 2), dtype=np.float32)
    wav, sr = read_c0_audio(audio, reader=_reader(stereo, 16000))
    assert sr == 16000
    assert wav.shape == (8,)
    assert wav.dtype == np.float32


def test_audio_rejects_duration_and_size(tmp_path: Path):
    audio = tmp_path / "long.wav"
    audio.write_bytes(b"x" * 8)
    too_long = np.ones(int(SAMPLE_RATE * 31), dtype=np.float32)
    with pytest.raises(C0AudioError, match="max_audio_duration_seconds"):
        read_c0_audio(audio, reader=_reader(too_long, 16000), max_duration_seconds=30)
    huge = tmp_path / "huge.wav"
    huge.write_bytes(b"x" * 64)
    with pytest.raises(C0AudioError, match="max_audio_bytes"):
        read_c0_audio(huge, reader=_reader(np.ones(16, dtype=np.float32), 16000), max_audio_bytes=8)


def test_prepare_asr_waveforms_matches_rq1_stereo_mean():
    stereo = np.array([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32)
    prepared = prepare_asr_waveforms([stereo])[0]
    assert prepared.dtype == np.float32
    np.testing.assert_allclose(prepared, np.array([2.0, 6.0], dtype=np.float32))


def test_missing_audio_and_checkpoint_files(tmp_path: Path):
    with pytest.raises(C0AudioError, match="not found"):
        read_c0_audio(tmp_path / "missing.wav")
    asr = tmp_path / "asr"
    asr.mkdir()
    (asr / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(C0ConfigError, match="incomplete"):
        assert_checkpoint_ready(asr, ASR_REQUIRED_FILES, label="ASR")
    with pytest.raises(C0ConfigError, match="cpu, cuda, or auto"):
        resolve_device("mps")


def test_pipeline_asr_then_mt_schema(tmp_path: Path):
    seen = {}
    wav = np.ones(16000, dtype=np.float32)

    def transcribe(model, processor, audio, sample_rate=16000):
        seen["asr"] = (model, processor, audio, sample_rate)
        return "bahnar transcript"

    def translate(model, tokenizer, asr_text, **kwargs):
        seen["mt"] = (model, tokenizer, asr_text, kwargs)
        return "bản dịch"

    cascade = _cascade(tmp_path, transcribe, translate, wav, verified_names=True)
    audio = tmp_path / "bahnar.wav"
    audio.write_bytes(b"placeholder")
    payload = cascade.run(audio)
    assert payload["model_id"] == MODEL_ID
    assert payload["asr_text_bahnar"] == "bahnar transcript"
    assert payload["translated_text_vi"] == "bản dịch"
    assert payload["duration_ms"] == 1000
    assert payload["latency_ms"] >= 0
    assert payload["checkpoint_identity"] == "expected_step_match"
    assert payload["research_provenance"] == "NOT_VERIFIED"
    assert payload["artifacts_present"] is True
    assert payload["generation_matches_rq1"] is True
    assert payload["asr_checkpoint_step"] == 12500
    assert payload["mt_checkpoint_step"] == 17500
    assert "asr_experiment_id" not in payload
    assert payload["tokenizer_source"] == "hub_pinned_rq1"
    assert "error" not in payload
    assert "confidence" not in payload
    assert "asr_model_id" not in payload
    assert seen["asr"][0] == "asr-model"
    assert isinstance(seen["asr"][2], np.ndarray)
    np.testing.assert_array_equal(seen["asr"][2], wav)
    assert seen["mt"][2] == "bahnar transcript"
    assert seen["mt"][3]["num_beams"] == 4

    asr_model = cascade.asr["model"]
    again = cascade.infer_waveform(wav, 16000)
    assert again["asr_text_bahnar"] == "bahnar transcript"
    assert cascade.asr["model"] is asr_model
    assert cascade.mt["tokenizer"] is cascade.mt["tokenizer"]


def test_pipeline_keeps_asr_text_when_mt_fails(tmp_path: Path):
    def translate(*_args, **_kwargs):
        raise C0ConfigError("mt failed")

    cascade = _cascade(
        tmp_path,
        lambda *_a, **_k: "xin",
        translate,
        np.ones(1600, dtype=np.float32),
    )
    audio = tmp_path / "bahnar.wav"
    audio.write_bytes(b"placeholder")
    payload = cascade.run(audio)
    assert payload["asr_text_bahnar"] == "xin"
    assert "translated_text_vi" not in payload
    assert payload["error"] == "mt failed"
    assert payload["error_code"] == "config_error"
    assert "duration_ms" in payload


def test_empty_and_whitespace_asr_does_not_call_mt(tmp_path: Path):
    audio = tmp_path / "bahnar.wav"
    audio.write_bytes(b"placeholder")
    wav = np.ones(1600, dtype=np.float32)
    for asr_text in ("", "   "):
        called = {"mt": False}

        def translate(*_a, **_k):
            called["mt"] = True
            return "should-not-run"

        cascade = _cascade(tmp_path, lambda *_a, **_k: asr_text, translate, wav)
        payload = cascade.run(audio)
        assert called["mt"] is False
        assert payload["error_code"] == "empty_asr"
        assert "translated_text_vi" not in payload
        assert "asr_text_bahnar" not in payload


def test_empty_and_whitespace_mt_is_serving_error(tmp_path: Path):
    audio = tmp_path / "bahnar.wav"
    audio.write_bytes(b"placeholder")
    wav = np.ones(1600, dtype=np.float32)
    for vi in ("", "\n  \t"):
        cascade = _cascade(tmp_path, lambda *_a, **_k: "xin", lambda *_a, **_k: vi, wav)
        payload = cascade.run(audio)
        assert payload["asr_text_bahnar"] == "xin"
        assert payload["error_code"] == "empty_mt"
        assert "translated_text_vi" not in payload


def test_research_bahnar_normalization_is_non_empty():
    for text in RESEARCH_BAHNAR:
        assert normalize_bahnar_ctc_v1(text)


def test_generation_config_error_and_no_runtime_path():
    with pytest.raises(C0ConfigError, match="num_beams"):
        C0Cascade(
            {
                "asr_checkpoint": "a",
                "mt_checkpoint": "b",
                "device": "cpu",
                "max_source_length": 0,
                "generation_max_length": 256,
                "num_beams": 4,
            },
            asr_bundle={"model": object(), "processor": object()},
            mt_bundle={"model": object(), "tokenizer": object()},
        )
    source = (ROOT / "src" / "c0_cascaded.py").read_text(encoding="utf-8")
    manifest = (ROOT / "src" / "c0_manifest.py").read_text(encoding="utf-8")
    tok = (ROOT / "src" / "c0_tokenizer.py").read_text(encoding="utf-8")
    assert "/tmp/bahnar-runtime" not in source
    assert "/tmp/bahnar-runtime" not in manifest
    assert "/tmp/bahnar-runtime" not in tok
    assert "/workspace/bahnar-s2tt-thesis" not in source
    for name in TRAINING_STATE_FILES:
        assert name not in ASR_REQUIRED_FILES
        assert name not in MT_REQUIRED_FILES


def test_asr_decode_contract_matches_rq1_source():
    rq1 = (ROOT / "src" / "rq1_inference.py").read_text(encoding="utf-8")
    c0 = (ROOT / "src" / "c0_cascaded.py").read_text(encoding="utf-8")
    assert 'processor(wavs, sampling_rate=16000, padding=True, return_tensors="pt")' in rq1
    assert ASR_PROCESSOR_KWARGS == {"sampling_rate": 16000, "padding": True, "return_tensors": "pt"}
    assert "torch.argmax(logits, dim=-1)" in rq1
    assert "torch.argmax(logits, dim=-1)" in c0
    assert "processor.batch_decode(pred_ids)" in rq1
    assert "processor.batch_decode(pred_ids)" in c0
    assert "normalize_bahnar_ctc_v1" in c0
    assert "from src.rq1_inference import run_asr_predictions" not in c0


def test_custom_checkpoint_does_not_inherit_verified_identity(tmp_path: Path):
    cascade = _cascade(
        tmp_path,
        lambda *_a, **_k: "xin",
        lambda *_a, **_k: "vi",
        np.ones(16, dtype=np.float32),
        verified_names=False,
    )
    audio = tmp_path / "bahnar.wav"
    audio.write_bytes(b"x")
    payload = cascade.run(audio)
    assert payload["checkpoint_identity"] == "custom_unverified"
    assert "asr_experiment_id" not in payload
    assert checkpoint_identity(Path("other"), Path("checkpoint-17500")) == "custom_unverified"
    assert checkpoint_identity(Path("/not-real/checkpoint-12500"), Path("/not-real/checkpoint-17500")) == "expected_step_match"
    assert checkpoint_step(Path("checkpoint-12500")) == 12500


def test_empty_config_checkpoint_is_missing_not_cwd():
    manifest = build_manifest(
        {
            "asr_checkpoint": "",
            "mt_checkpoint": "",
            "device": "cpu",
            "max_source_length": 256,
            "generation_max_length": 256,
            "num_beams": 4,
        }
    )
    assert manifest["checkpoint_identity"] == "missing"
    assert manifest["fine_tuned"]["asr"]["identity"] == "missing"
    assert manifest["fine_tuned"]["asr"]["checkpoint"] is None
    assert manifest["fine_tuned"]["mt"]["identity"] == "missing"


def test_manifest_separates_pretrained_and_finetuned(tmp_path: Path):
    manifest = build_manifest(_config(tmp_path, verified_names=True), compute_sha256=False)
    assert manifest["pretrained"]["asr"]["pretrained_model_id"] == "facebook/wav2vec2-xls-r-300m"
    assert manifest["pretrained"]["mt"]["pretrained_model_id"] == "vinai/bartpho-syllable"
    assert manifest["fine_tuned"]["asr"]["checkpoint_step"] == 12500
    assert manifest["fine_tuned"]["asr"]["identity"] == "expected_step_match"
    assert manifest["fine_tuned"]["asr"]["artifacts_present"] is True
    assert manifest["fine_tuned"]["asr"]["experiment_id"] is None
    assert manifest["checkpoint_identity"] == "expected_step_match"
    assert manifest["decoding"]["generation"]["matches_rq1"] is True
    assert manifest["fine_tuned"]["asr"]["weights_sha256"] is None
    assert manifest["fine_tuned"]["asr"]["weights_sha256_status"] == "NOT_COMPUTED"
    assert manifest["tokenizer"]["tokenizer_source"] == "hub_pinned_rq1"
    assert manifest["tokenizer"]["tokenizer_parity"] == "NOT_VERIFIED"
    assert manifest["runtime"]["required_dependencies"]["torch"] == "2.8.0"
    assert "torch" in manifest["runtime"]["installed_dependencies"]
    assert manifest["audio"]["serving"]["max_audio_duration_seconds"] == 30
    assert DEFAULT_MAX_AUDIO_DURATION_SECONDS == 30.0
    custom = build_manifest(_config(tmp_path, verified_names=False), compute_sha256=True)
    assert custom["fine_tuned"]["mt"]["identity"] == "custom_unverified"
    assert custom["fine_tuned"]["mt"]["experiment_id"] is None
    assert custom["fine_tuned"]["mt"]["weights_sha256_status"] == "COMPUTED"
    assert len(custom["fine_tuned"]["mt"]["weights_sha256"]) == 64


def test_yaml_config_errors_are_structured(tmp_path: Path, capsys):
    missing = tmp_path / "missing.yaml"
    bad = tmp_path / "bad.yaml"
    bad.write_text("{invalid", encoding="utf-8")
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    listed = tmp_path / "list.yaml"
    listed.write_text("- a\n- b\n", encoding="utf-8")
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    for path in (missing, bad, empty, listed):
        with pytest.raises(C0ConfigError):
            load_c0_config(path)
        code = c0_main(["--config", str(path), "--audio", str(audio)])
        out = capsys.readouterr().out
        payload = __import__("json").loads(out)
        assert code == 1
        assert payload["error_code"] == "config_error"
        assert "error" in payload
    manifest_code = manifest_main(["--config", str(bad)])
    manifest_out = capsys.readouterr().out
    assert manifest_code == 1
    assert __import__("json").loads(manifest_out)["error_code"] == "config_error"
    as_dir = tmp_path / "configdir"
    as_dir.mkdir()
    with pytest.raises(C0ConfigError, match="Cannot read config"):
        load_c0_config(as_dir)


def test_name_only_checkpoint_is_not_verified():
    ident = checkpoint_identity(Path("/not-real/checkpoint-12500"), Path("/not-real/checkpoint-17500"))
    assert ident == "expected_step_match"
    asr = assess_checkpoint(Path("/not-real/checkpoint-12500"), ASR_REQUIRED_FILES, expected_step=12500)
    assert asr["artifacts_present"] is False
    assert asr["identity"] == "expected_step_match"


def test_operator_yaml_hash_is_not_verified_c0(tmp_path: Path):
    import hashlib

    digest = hashlib.sha256(b"not-a-weight").hexdigest()
    cascade = _cascade(
        tmp_path,
        lambda *_a, **_k: "xin",
        lambda *_a, **_k: "vi",
        np.ones(16, dtype=np.float32),
        verified_names=True,
        extra={"asr_weights_sha256": digest, "mt_weights_sha256": digest, "research_pins": ""},
    )
    audio = tmp_path / "bahnar.wav"
    audio.write_bytes(b"x")
    payload = cascade.run(audio)
    assert payload["checkpoint_identity"] == "configured_hash_match"
    assert payload["research_provenance"] == "NOT_VERIFIED"
    assert "asr_experiment_id" not in payload
    assert payload["artifacts_present"] is True


def test_fake_checkpoint_and_fake_research_pins_are_not_verified(tmp_path: Path):
    import hashlib

    digest = hashlib.sha256(b"not-a-weight").hexdigest()
    pins = tmp_path / "c0_research_pins.yaml"
    pins.write_text(f"asr_weights_sha256: {digest}\nmt_weights_sha256: {digest}\n", encoding="utf-8")
    cascade = _cascade(
        tmp_path,
        lambda *_a, **_k: "xin",
        lambda *_a, **_k: "vi",
        np.ones(16, dtype=np.float32),
        verified_names=True,
        extra={"research_pins": str(pins)},
    )
    audio = tmp_path / "bahnar.wav"
    audio.write_bytes(b"x")
    payload = cascade.run(audio)
    assert payload["checkpoint_identity"] == "research_reference_hash_match"
    assert payload["research_provenance"] == "REFERENCE_HASH_MATCH"
    assert payload["checkpoint_identity"] != "verified_c0"
    assert payload["research_provenance"] != "VERIFIED"
    assert "asr_experiment_id" not in payload


def test_research_reference_match_is_not_independent_verification(tmp_path: Path):
    import hashlib

    digest = hashlib.sha256(b"not-a-weight").hexdigest()
    asr = tmp_path / "checkpoint-12500"
    _touch_checkpoint(asr, ASR_REQUIRED_FILES)
    assessed = assess_checkpoint(
        asr,
        ASR_REQUIRED_FILES,
        expected_step=12500,
        research_sha256=digest,
    )
    assert assessed["research_hash_match"] is True
    assert assessed["identity"] == "research_reference_hash_match"
    assert assessed["research_provenance"] == "REFERENCE_HASH_MATCH"
    assert assessed["identity"] != "verified_c0"
    assert assessed["research_provenance"] != "VERIFIED"


def test_research_pin_mismatch_is_not_verified_or_reference_match(tmp_path: Path):
    pins = tmp_path / "c0_research_pins.yaml"
    pins.write_text(f"asr_weights_sha256: {'0' * 64}\nmt_weights_sha256: {'0' * 64}\n", encoding="utf-8")
    cascade = _cascade(
        tmp_path,
        lambda *_a, **_k: "xin",
        lambda *_a, **_k: "vi",
        np.ones(16, dtype=np.float32),
        verified_names=True,
        extra={"research_pins": str(pins)},
    )
    audio = tmp_path / "bahnar.wav"
    audio.write_bytes(b"x")
    payload = cascade.run(audio)
    assert payload["checkpoint_identity"] != "verified_c0"
    assert payload["checkpoint_identity"] != "research_reference_hash_match"
    assert payload["research_provenance"] == "NOT_VERIFIED"
    assert payload["research_provenance"] != "REFERENCE_HASH_MATCH"
    assert "asr_experiment_id" not in payload


def test_wrong_operator_hash_fails_closed_before_load(tmp_path: Path, monkeypatch):
    called = {"asr": False}

    def boom(*_a, **_k):
        called["asr"] = True
        raise AssertionError("model load must not run after checksum mismatch")

    monkeypatch.setattr("src.c0_cascaded.load_asr_bundle", boom)
    monkeypatch.setattr("src.c0_cascaded.load_mt_bundle", boom)
    cfg = _config(tmp_path)
    cfg["asr_weights_sha256"] = "0" * 64
    cfg["mt_weights_sha256"] = "0" * 64
    with pytest.raises(C0ChecksumError, match="configured pin") as excinfo:
        C0Cascade(cfg)
    assert called["asr"] is False
    assert excinfo.value.error_code == "checkpoint_checksum_mismatch"


def test_invalid_sha_format_is_config_error(tmp_path: Path):
    with pytest.raises(C0ConfigError, match="64-character"):
        _cascade(
            tmp_path,
            lambda *_a, **_k: "xin",
            lambda *_a, **_k: "vi",
            np.ones(16, dtype=np.float32),
            extra={"asr_weights_sha256": "not-a-hash"},
        )


def test_unreadable_weights_fail_checksum(tmp_path: Path, monkeypatch):
    cfg = _config(tmp_path)
    cfg["asr_weights_sha256"] = "a" * 64

    def bad_hash(_path):
        raise OSError("permission denied")

    monkeypatch.setattr("src.c0_cascaded.sha256_file", bad_hash)
    with pytest.raises(C0ChecksumError, match="not readable") as excinfo:
        C0Cascade(
            cfg,
            asr_bundle={"model": object(), "processor": object(), "checkpoint": cfg["asr_checkpoint"]},
            mt_bundle={"model": object(), "tokenizer": object(), "checkpoint": cfg["mt_checkpoint"]},
        )
    assert excinfo.value.error_code == "checkpoint_weights_unreadable"


def test_missing_checkpoint_wrong_step_and_missing_weights_are_not_verified(tmp_path: Path):
    missing = assess_checkpoint(None, ASR_REQUIRED_FILES, expected_step=12500)
    assert missing["identity"] == "missing"
    assert missing["research_provenance"] == "NOT_VERIFIED"

    absent = assess_checkpoint(tmp_path / "checkpoint-12500", ASR_REQUIRED_FILES, expected_step=12500)
    assert absent["artifacts_present"] is False
    assert absent["identity"] != "verified_c0"
    assert absent["research_provenance"] == "NOT_VERIFIED"

    wrong = tmp_path / "checkpoint-1"
    _touch_checkpoint(wrong, ASR_REQUIRED_FILES)
    wrong_assessed = assess_checkpoint(wrong, ASR_REQUIRED_FILES, expected_step=12500)
    assert wrong_assessed["step_name_match"] is False
    assert wrong_assessed["identity"] == "custom_unverified"
    assert wrong_assessed["identity"] != "verified_c0"

    incomplete = tmp_path / "checkpoint-12500"
    incomplete.mkdir()
    (incomplete / "config.json").write_text("{}", encoding="utf-8")
    incomplete_assessed = assess_checkpoint(incomplete, ASR_REQUIRED_FILES, expected_step=12500)
    assert incomplete_assessed["artifacts_present"] is False
    assert incomplete_assessed["identity"] != "verified_c0"
    assert "model.safetensors" in incomplete_assessed["missing_files"]


def test_provenance_does_not_depend_on_bahnar_runtime_root(tmp_path: Path):
    asr = tmp_path / "checkpoint-12500"
    mt = tmp_path / "checkpoint-17500"
    _touch_checkpoint(asr, ASR_REQUIRED_FILES)
    _touch_checkpoint(mt, MT_REQUIRED_FILES)
    asr_assessed = assess_checkpoint(asr, ASR_REQUIRED_FILES, expected_step=12500)
    mt_assessed = assess_checkpoint(mt, MT_REQUIRED_FILES, expected_step=17500)
    assert str(tmp_path) in asr_assessed["checkpoint"]
    assert "/tmp/bahnar-runtime" not in asr_assessed["checkpoint"]
    assert "/tmp/bahnar-runtime" not in mt_assessed["checkpoint"]
    assert asr_assessed["identity"] == "expected_step_match"
    assert asr_assessed["research_provenance"] == "NOT_VERIFIED"


def test_configured_hash_integrity_pass_is_not_verified_c0(tmp_path: Path):
    import hashlib

    digest = hashlib.sha256(b"not-a-weight").hexdigest()
    asr = tmp_path / "checkpoint-12500"
    _touch_checkpoint(asr, ASR_REQUIRED_FILES)
    assessed = assess_checkpoint(
        asr,
        ASR_REQUIRED_FILES,
        expected_step=12500,
        configured_sha256=digest,
    )
    assert assessed["configured_hash_match"] is True
    assert assessed["identity"] == "configured_hash_match"
    assert assessed["research_provenance"] == "NOT_VERIFIED"


def test_missing_weights_fail_checksum_before_load(tmp_path: Path, monkeypatch):
    cfg = _config(tmp_path)
    (Path(cfg["asr_checkpoint"]) / "model.safetensors").unlink()
    cfg["asr_weights_sha256"] = "a" * 64
    called = {"asr": False}

    def boom(*_a, **_k):
        called["asr"] = True
        raise AssertionError("model load must not run after missing weights")

    monkeypatch.setattr("src.c0_cascaded.load_asr_bundle", boom)
    with pytest.raises((C0ConfigError, C0ChecksumError)):
        C0Cascade(cfg)
    assert called["asr"] is False


def test_cli_checksum_mismatch_exits_nonzero_json(tmp_path: Path, capsys, monkeypatch):
    import json

    import yaml

    cfg = _config(tmp_path, verified_names=True)
    cfg["asr_weights_sha256"] = "0" * 64
    yaml_path = tmp_path / "c0.yaml"
    yaml_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")

    def boom(*_a, **_k):
        raise AssertionError("model load must not run after checksum mismatch")

    monkeypatch.setattr("src.c0_cascaded.load_asr_bundle", boom)
    monkeypatch.setattr("src.c0_cascaded.load_mt_bundle", boom)
    code = c0_main(["--config", str(yaml_path), "--audio", str(audio)])
    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["error_code"] == "checkpoint_checksum_mismatch"
    assert "error" in payload


def test_manifest_cli_invalid_max_source_length(tmp_path: Path, capsys):
    import json

    import yaml

    cfg = _config(tmp_path)
    cfg["max_source_length"] = "invalid"
    yaml_path = tmp_path / "c0.yaml"
    yaml_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    code = manifest_main(["--config", str(yaml_path)])
    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["error_code"] == "config_error"
    assert "error" in payload


def test_manifest_cli_creates_missing_out_parent(tmp_path: Path):
    import json

    import yaml

    cfg = _config(tmp_path, verified_names=True)
    yaml_path = tmp_path / "c0.yaml"
    yaml_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    out = tmp_path / "nested" / "dir" / "manifest.json"
    code = manifest_main(["--config", str(yaml_path), "--out", str(out)])
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["model_id"] == MODEL_ID
    assert payload["research_provenance"] == "NOT_VERIFIED"
    assert payload["checkpoint_identity"] == "expected_step_match"


def test_manifest_cli_unwritable_out_is_output_error(tmp_path: Path, monkeypatch, capsys):
    import json

    import yaml

    cfg = _config(tmp_path)
    yaml_path = tmp_path / "c0.yaml"
    yaml_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    out = tmp_path / "out.json"
    real_write = Path.write_text

    def boom(self, *args, **kwargs):
        if Path(self) == out:
            raise OSError("read-only filesystem")
        return real_write(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", boom)
    code = manifest_main(["--config", str(yaml_path), "--out", str(out)])
    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["error_code"] == "output_error"
    assert "error" in payload


def test_nonfinite_duration_is_rejected(tmp_path: Path):
    cfg = _config(tmp_path)
    bundles = dict(
        asr_bundle={"model": object(), "processor": object()},
        mt_bundle={"model": object(), "tokenizer": object()},
    )
    for value in (float("inf"), float("nan"), float("-inf")):
        cfg["max_audio_duration_seconds"] = value
        with pytest.raises(C0ConfigError, match="finite"):
            C0Cascade(cfg, **bundles)


def test_generation_override_is_not_rq1_contract(tmp_path: Path):
    cascade = _cascade(
        tmp_path,
        lambda *_a, **_k: "xin",
        lambda *_a, **_k: "vi",
        np.ones(16, dtype=np.float32),
        verified_names=True,
        extra={"max_source_length": 128, "generation_max_length": 64, "num_beams": 1},
    )
    audio = tmp_path / "bahnar.wav"
    audio.write_bytes(b"x")
    payload = cascade.run(audio)
    assert payload["generation_matches_rq1"] is False
    assert payload["checkpoint_identity"] == "expected_step_match"
    assert "asr_experiment_id" not in payload
    manifest = build_manifest({**_config(tmp_path, verified_names=True), "max_source_length": 128, "generation_max_length": 64, "num_beams": 1})
    assert manifest["decoding"]["generation"]["matches_rq1"] is False


def test_cli_emits_json_when_model_import_fails(tmp_path: Path, capsys, monkeypatch):
    import json

    import yaml

    cfg = _config(tmp_path)
    yaml_path = tmp_path / "c0.yaml"
    yaml_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")

    def boom(*_a, **_k):
        raise ImportError("No module named 'transformers'")

    monkeypatch.setattr("src.c0_cascaded.load_asr_bundle", boom)
    code = c0_main(["--config", str(yaml_path), "--audio", str(audio)])
    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["error_code"] == "dependency_missing"


def test_missing_generation_key_is_config_error(tmp_path: Path):
    cfg = _config(tmp_path)
    del cfg["num_beams"]
    with pytest.raises(C0ConfigError, match="generation config"):
        C0Cascade(
            cfg,
            asr_bundle={"model": object(), "processor": object()},
            mt_bundle={"model": object(), "tokenizer": object()},
        )


def test_inference_manifest_and_cli_agree_on_reference_hash_match(tmp_path: Path, capsys):
    import hashlib
    import json

    import yaml

    digest = hashlib.sha256(b"not-a-weight").hexdigest()
    pins = tmp_path / "c0_research_pins.yaml"
    pins.write_text(f"asr_weights_sha256: {digest}\nmt_weights_sha256: {digest}\n", encoding="utf-8")
    cfg = _config(tmp_path, verified_names=True)
    cfg["research_pins"] = str(pins)
    cascade = _cascade(
        tmp_path,
        lambda *_a, **_k: "xin",
        lambda *_a, **_k: "vi",
        np.ones(16, dtype=np.float32),
        verified_names=True,
        extra={"research_pins": str(pins)},
    )
    audio = tmp_path / "bahnar.wav"
    audio.write_bytes(b"x")
    inference = cascade.run(audio)
    manifest = build_manifest(cfg)
    yaml_path = tmp_path / "c0.yaml"
    yaml_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    out = tmp_path / "manifest.json"
    code = manifest_main(["--config", str(yaml_path), "--out", str(out)])
    assert code == 0
    cli_manifest = json.loads(out.read_text(encoding="utf-8"))
    for payload in (inference, manifest, cli_manifest):
        assert payload["checkpoint_identity"] == "research_reference_hash_match"
        assert payload["research_provenance"] == "REFERENCE_HASH_MATCH"
        assert payload["checkpoint_identity"] != "verified_c0"
        assert payload["research_provenance"] != "VERIFIED"
        assert payload.get("asr_experiment_id") in (None, )
        assert "asr_experiment_id" not in inference
    assert manifest["fine_tuned"]["asr"]["experiment_id"] is None
    assert manifest["fine_tuned"]["asr"]["research_provenance"] == "REFERENCE_HASH_MATCH"
    capsys.readouterr()


def test_c0_source_does_not_assign_false_verified_labels():
    cascaded = (ROOT / "src" / "c0_cascaded.py").read_text(encoding="utf-8")
    manifest = (ROOT / "src" / "c0_manifest.py").read_text(encoding="utf-8")
    assert 'identity = "verified_c0"' not in cascaded
    assert 'research_provenance = "VERIFIED"' not in cascaded
    assert 'return "verified_c0"' not in cascaded
    assert '"VERIFIED"' not in manifest or "tokenizer_parity" in manifest
    assert 'research_provenance": (\n            "VERIFIED"' not in manifest
