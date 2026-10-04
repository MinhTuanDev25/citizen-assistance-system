"""Notebook 12 pseudo-label inference tests (synthetic only; no real model or artifact)."""
from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.rq1_contract import sha256_json
from src.rq2_pseudo_contract import (
    DataAccessLedger,
    GTestAccessError,
    Nb11InputError,
    TeacherContractError,
    TeacherDecodingConfig,
    assert_nb11_input_unchanged,
    assert_nb12_output_dir,
    assert_not_g_test_path,
    build_d0_agreement_contract,
    build_teacher_contract,
    decoding_config_from_rq1,
    resolve_fixed_teacher,
    resolve_nb11_input,
    verify_nb11_segment_audio,
)
from src.rq2_pseudo_label import (
    FORBIDDEN_REFERENCE_FIELDS,
    RAW_RECORD_COLUMNS,
    STATUS_DISABLED,
    STATUS_FAILED,
    STATUS_OK,
    STATUS_SKIPPED_ASR_EMPTY,
    AudioIdentityError,
    AudioItem,
    CheckpointBindingError,
    HfCtcAsrAdapter,
    HfDirectD0Adapter,
    HfSeq2SeqMtAdapter,
    InferenceCheckpoint,
    ItemInferenceError,
    ReferenceLeakError,
    agreement_features,
    archive_checkpoint,
    ctc_greedy_confidence,
    inference_binding,
    items_from_audio_frame,
    items_from_nb11,
    iter_raw_records,
    load_verified_waveform,
    materialize_validation_audio,
    run_asr_stage,
    run_mt_stage,
    run_peak_safe_teacher_inference,
    seq2seq_mean_logprob,
    text_is_valid,
)
from tests.rq2_nb12_fixtures import build_nb11_generation, fake_contracts, fake_rq1_chain, tree_digest

ROOT = Path(__file__).resolve().parents[1]
NB12_SOURCES = [ROOT / "src" / "rq2_pseudo_contract.py", ROOT / "src" / "rq2_pseudo_label.py", ROOT / "src" / "rq2_quality.py"]
NB12_NOTEBOOK = ROOT / "notebooks" / "12_RQ2_PseudoLabel_Quality_Freeze_UPrime.ipynb"


# --------------------------------------------------------------------------- #
# Fake teacher adapters                                                        #
# --------------------------------------------------------------------------- #
def fake_asr(waves):
    out = []
    for w in waves:
        n = len(w)
        text = "" if n == 800 else f"bah{n % 7} nar{int(abs(w).sum()) % 5}"
        lp = -0.01 * (n % 13) - 0.05
        out.append({"text_raw": text, "asr_token_count": len(text), "asr_mean_logprob": lp if text else None,
                    "asr_confidence_raw": math.exp(lp) if text else None, "asr_frame_mean_logprob": lp,
                    "asr_blank_frame_fraction": 0.5, "asr_n_frames": n // 320})
    return out


def fake_mt(texts):
    return [{"text_raw": f"vi {t}", "mt_token_count": len(t), "mt_mean_logprob": -0.1 * (len(t) % 5) - 0.1,
             "mt_confidence_raw": 0.5, "source_truncated": False, "hit_max_length": False} for t in texts]


def fake_d0(waves):
    return [{"text_raw": f"vi bah{len(w) % 7}"} for w in waves]


def _pool(tmp_path, *, d0_enabled=False, shard_size=3, n_segments=7, n_samples=()):
    nb = build_nb11_generation(tmp_path, n_segments=n_segments, n_samples=n_samples)
    nb11 = resolve_nb11_input(tmp_path)
    teacher, d0, decoding, _, _ = fake_contracts(d0_enabled=d0_enabled)
    binding = inference_binding(
        pool="u_clean", input_identity_sha256=nb11.contract_sha256,
        teacher_contract_sha256=teacher["teacher_contract_sha256"],
        d0_agreement_contract_sha256=d0["d0_agreement_contract_sha256"],
        decoding_config_sha256=teacher["decoding_config_sha256"],
    )
    identity = {"nb11_generation_id": nb11.generation_id, "nb11_input_contract_sha256": nb11.contract_sha256,
                "teacher_contract_sha256": teacher["teacher_contract_sha256"],
                "d0_agreement_contract_sha256": d0["d0_agreement_contract_sha256"]}
    decoding_payload = teacher["decoding"] if not d0_enabled else {**teacher["decoding"], "d0": d0["decoding"]}
    return SimpleNamespace(nb=nb, nb11=nb11, teacher=teacher, d0=d0, binding=binding, identity=identity,
                           decoding=decoding_payload, items=items_from_nb11(nb11), shard_size=shard_size)


def _run(pool, ckpt_root, *, asr=fake_asr, mt=fake_mt, d0=fake_d0):
    ckpt = InferenceCheckpoint(ckpt_root, binding=pool.binding, ordered_uids=pool.nb11.ordered_uids, shard_size=pool.shard_size)
    released = []

    def loader(name, fn):
        return lambda: {"adapter": fn, "release": lambda: released.append(name)}

    report = run_peak_safe_teacher_inference(
        ckpt=ckpt, items=pool.items, decoding=pool.decoding,
        load_asr=loader("asr", asr), load_mt=loader("mt", mt),
        load_d0=loader("d0", d0) if "d0" in pool.decoding else None,
    )
    records = list(iter_raw_records(ckpt, pool.nb11.rows, identity=pool.identity, d0_enabled="d0" in pool.decoding))
    return ckpt, report, records, released


# --------------------------------------------------------------------------- #
# A. NB11 input contract                                                       #
# --------------------------------------------------------------------------- #
def test_resolve_nb11_input_valid_and_stable(tmp_path):
    build_nb11_generation(tmp_path)
    a = resolve_nb11_input(tmp_path)
    b = resolve_nb11_input(tmp_path)
    assert a.contract_sha256 == b.contract_sha256
    assert a.contract["n_segments"] == 7
    assert verify_nb11_segment_audio(a)["n_verified"] == 7


def test_nb11_absent_refuses(tmp_path):
    with pytest.raises(Nb11InputError, match="no published generation"):
        resolve_nb11_input(tmp_path)


def test_nb11_incomplete_generation_refuses(tmp_path):
    build_nb11_generation(tmp_path, write_complete=False)
    with pytest.raises(Nb11InputError, match="incomplete"):
        resolve_nb11_input(tmp_path)


def test_nb11_wrong_status_refuses(tmp_path):
    build_nb11_generation(tmp_path, status="FAIL_RQ2_U_CLEAN")
    with pytest.raises(Nb11InputError, match="status"):
        resolve_nb11_input(tmp_path)


def test_nb11_segmentation_pin_drift_refuses(tmp_path):
    build_nb11_generation(tmp_path, segmentation_sha="0" * 64)
    with pytest.raises(Nb11InputError, match="segmentation_contract_sha256"):
        resolve_nb11_input(tmp_path)


def test_nb11_duplicate_uid_refuses(tmp_path):
    build_nb11_generation(tmp_path, duplicate_uid=True)
    with pytest.raises(Nb11InputError, match="duplicate segment_uid"):
        resolve_nb11_input(tmp_path)


def test_nb11_schema_drift_refuses(tmp_path):
    build_nb11_generation(tmp_path, extra_column=True)
    with pytest.raises(Nb11InputError, match="schema drift"):
        resolve_nb11_input(tmp_path)


def test_nb11_audio_mismatch_is_fatal(tmp_path):
    nb = build_nb11_generation(tmp_path)
    nb11 = resolve_nb11_input(tmp_path)
    wav = tmp_path / nb["rows"][2]["segment_local_path"]
    data = bytearray(wav.read_bytes())
    data[-2:] = b"\x11\x22"
    wav.write_bytes(bytes(data))
    with pytest.raises(Nb11InputError, match="hash differs"):
        verify_nb11_segment_audio(nb11)
    with pytest.raises(AudioIdentityError):
        load_verified_waveform(items_from_nb11(nb11)[2])


def test_nb11_change_after_lock_fails_closed(tmp_path):
    nb = build_nb11_generation(tmp_path)
    locked = resolve_nb11_input(tmp_path).contract
    summary = nb["gen_dir"] / "summary.json"
    payload = json.loads(summary.read_text())
    payload["note"] = "edited"
    summary.write_text(json.dumps(payload))
    with pytest.raises(Nb11InputError, match="changed since it was locked"):
        assert_nb11_input_unchanged(locked, resolve_nb11_input(tmp_path))


def test_nb11_checkpoint_dir_is_never_read(tmp_path):
    nb = build_nb11_generation(tmp_path)
    ckpt = nb["out_dir"] / "checkpoint"
    ckpt.mkdir()
    (ckpt / "u_clean_manifest.jsonl").write_text("not json at all\n")
    assert resolve_nb11_input(tmp_path).contract["n_segments"] == 7


# --------------------------------------------------------------------------- #
# B. Teacher contract                                                          #
# --------------------------------------------------------------------------- #
def test_teacher_contract_deterministic_and_bound_to_checkpoint():
    t1 = fake_contracts()[0]
    t2 = fake_contracts()[0]
    t3 = fake_contracts(digest="4" * 64)[0]
    assert t1["teacher_contract_sha256"] == t2["teacher_contract_sha256"]
    assert t1["teacher_contract_sha256"] != t3["teacher_contract_sha256"]
    assert t1["role"] == "fixed_rq1_c0_teacher"
    assert "metrics" not in json.dumps(t1)


def test_decoding_change_changes_teacher_hash():
    final_contract, handoff = fake_rq1_chain()
    a = build_teacher_contract(final_contract=final_contract, handoff=handoff, decoding=TeacherDecodingConfig(mt_batch_size=16))
    b = build_teacher_contract(final_contract=final_contract, handoff=handoff, decoding=TeacherDecodingConfig(mt_batch_size=8))
    assert a["teacher_contract_sha256"] != b["teacher_contract_sha256"]
    with pytest.raises(ValueError):
        TeacherDecodingConfig(precision="fp16").payload(handoff["mt"]["training_contract"], None)


def test_decoding_batch_sizes_come_from_rq1_contracts():
    _, handoff = fake_rq1_chain(mt_batch=12)
    cfg = decoding_config_from_rq1(handoff["mt"]["training_contract"], handoff["direct"]["training_contract"])
    assert (cfg.asr_batch_size, cfg.mt_batch_size, cfg.d0_batch_size) == (1, 12, 2)


def test_teacher_handoff_mismatch_and_unverified_proof_refuse():
    final_contract, handoff = fake_rq1_chain()
    bad = {**handoff, "mt_handoff_hash": "0" * 64}
    with pytest.raises(TeacherContractError, match="mt_handoff_hash"):
        build_teacher_contract(final_contract=final_contract, handoff=bad, decoding=TeacherDecodingConfig())
    fc = json.loads(json.dumps(final_contract))
    fc["checkpoint_proof"]["asr"]["verified"] = False
    with pytest.raises(TeacherContractError, match="not verified"):
        build_teacher_contract(final_contract=fc, handoff=handoff, decoding=TeacherDecodingConfig())


def test_d0_disabled_vs_enabled_contracts():
    final_contract, handoff = fake_rq1_chain()
    off = build_d0_agreement_contract(enabled=False)
    on = build_d0_agreement_contract(enabled=True, final_contract=final_contract, handoff=handoff, decoding=TeacherDecodingConfig())
    assert off["enabled"] is False and on["enabled"] is True
    assert on["role"] == "agreement_feature_only_not_a_teacher"
    assert off["d0_agreement_contract_sha256"] != on["d0_agreement_contract_sha256"]


def _write_final_state(tmp_path, final_contract, *, name=None, status="SUCCESS_RQ1_FINAL"):
    h = final_contract["rq1_final_contract_hash"]
    state = tmp_path / (name or f"contract_{h[:16]}")
    state.mkdir(parents=True)
    (state / "rq1_final_contract.json").write_text(json.dumps(final_contract))
    (state / "rq1_final_summary.json").write_text(json.dumps(
        {"status": status, "ready_rq1_final": True, "rq1_final_contract_hash": h, "metrics": {"c0_chrfpp": 1.0}}))
    return state


def test_resolve_fixed_teacher_rejects_misnamed_state_dir(tmp_path):
    final_contract, _ = fake_rq1_chain()
    state = _write_final_state(tmp_path, final_contract, name="contract_latest")
    with pytest.raises(TeacherContractError, match="state dir name"):
        resolve_fixed_teacher(project_root=ROOT, rq1_final_state_dir=state, asr_state_dir=tmp_path,
                              mt_state_dir=tmp_path, direct_state_dir=tmp_path,
                              decoding=TeacherDecodingConfig(), d0_agreement_enabled=False)


def test_resolve_fixed_teacher_rejects_tampered_or_unfinished_rq1(tmp_path):
    final_contract, _ = fake_rq1_chain()
    tampered = {**final_contract, "asr_handoff_hash": "0" * 64}
    state = _write_final_state(tmp_path / "a", tampered)
    with pytest.raises(TeacherContractError, match="self-hash"):
        resolve_fixed_teacher(project_root=ROOT, rq1_final_state_dir=state, asr_state_dir=tmp_path,
                              mt_state_dir=tmp_path, direct_state_dir=tmp_path,
                              decoding=TeacherDecodingConfig(), d0_agreement_enabled=False)
    state = _write_final_state(tmp_path / "b", final_contract, status="FAIL_RQ1_FINAL")
    with pytest.raises(TeacherContractError, match="SUCCESS_RQ1_FINAL"):
        resolve_fixed_teacher(project_root=ROOT, rq1_final_state_dir=state, asr_state_dir=tmp_path,
                              mt_state_dir=tmp_path, direct_state_dir=tmp_path,
                              decoding=TeacherDecodingConfig(), d0_agreement_enabled=False)


def test_resolve_fixed_teacher_rejects_source_drift(tmp_path):
    final_contract, _ = fake_rq1_chain()
    state = _write_final_state(tmp_path, final_contract)
    with pytest.raises(TeacherContractError, match="source code changed"):
        resolve_fixed_teacher(project_root=ROOT, rq1_final_state_dir=state, asr_state_dir=tmp_path,
                              mt_state_dir=tmp_path, direct_state_dir=tmp_path,
                              decoding=TeacherDecodingConfig(), d0_agreement_enabled=False)


def test_no_checkpoint_selection_heuristics_in_nb12_sources():
    for path in NB12_SOURCES:
        text = path.read_text(encoding="utf-8")
        for banned in ("getmtime", "st_mtime", "LATEST", "latest_checkpoint", "sorted(glob", "max(glob"):
            assert banned not in text, f"{banned} in {path.name}"


# --------------------------------------------------------------------------- #
# C. Pseudo inference + HF adapter parity                                      #
# --------------------------------------------------------------------------- #
def test_full_pool_one_record_per_uid_and_deterministic(tmp_path):
    pool = _pool(tmp_path, n_samples=(1600, 800, 1760, 1920, 2080, 2240, 2400))
    before = tree_digest(pool.nb["out_dir"])
    _, report, a, released = _run(pool, tmp_path / "ckpt_a")
    _, _, b, _ = _run(pool, tmp_path / "ckpt_b")
    assert [r["segment_uid"] for r in a] == pool.nb11.ordered_uids
    assert sha256_json(a) == sha256_json(b)
    assert released == ["asr", "mt"]
    assert all(list(r.keys()) == RAW_RECORD_COLUMNS for r in a)
    empty = a[1]
    assert empty["asr_empty"] and empty["mt_status"] == STATUS_SKIPPED_ASR_EMPTY
    assert a[0]["pseudo_vi_norm"].startswith("vi ") and a[0]["d0_status"] == STATUS_DISABLED
    assert tree_digest(pool.nb["out_dir"]) == before


def test_item_failure_marks_only_that_item(tmp_path):
    pool = _pool(tmp_path)
    ckpt = InferenceCheckpoint(tmp_path / "ckpt", binding=pool.binding, ordered_uids=pool.nb11.ordered_uids, shard_size=3)

    def flaky(waves):
        if any(len(w) == 1600 + 160 * 4 for w in waves):
            if len(waves) == 1:
                raise ItemInferenceError("decoder exploded")
            raise ItemInferenceError("batch failed")
        return fake_asr(waves)

    run_asr_stage(ckpt, pool.items, flaky, batch_size=8)
    run_mt_stage(ckpt, fake_mt, batch_size=8)
    records = list(iter_raw_records(ckpt, pool.nb11.rows, identity=pool.identity, d0_enabled=False))
    assert records[4]["asr_status"] == STATUS_FAILED and "decoder exploded" in records[4]["asr_decode_warning"]
    assert records[4]["mt_status"] != STATUS_OK
    assert all(r["asr_status"] == STATUS_OK for i, r in enumerate(records) if i != 4)


@pytest.mark.parametrize("exc", [ValueError("tensor shape mismatch"), RuntimeError("tokenizer config missing")])
def test_unknown_inference_error_raises_and_publishes_nothing(tmp_path, exc):
    pool = _pool(tmp_path)
    out = tmp_path / "artifacts" / "rq2" / "pseudo_labels"
    out.mkdir(parents=True)
    current = out / "CURRENT"
    current.write_text("sentinel-generation\n")

    def boom(waves):
        raise exc

    with pytest.raises(type(exc), match=str(exc)):
        _run(pool, out / "checkpoint", asr=boom)
    assert current.read_text() == "sentinel-generation\n"
    assert not (out / "generations").exists()


def test_adapter_output_count_mismatch_is_fatal(tmp_path):
    pool = _pool(tmp_path)

    def wrong_count(waves):
        return fake_asr(waves) + [{"text_raw": "extra"}]

    with pytest.raises(RuntimeError, match="different number of outputs"):
        _run(pool, tmp_path / "ckpt", asr=wrong_count)


def test_out_of_memory_is_fatal_not_an_exclusion(tmp_path):
    pool = _pool(tmp_path)

    def oom(waves):
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="out of memory"):
        _run(pool, tmp_path / "ckpt", asr=oom)


def test_audio_mismatch_during_inference_is_fatal(tmp_path):
    pool = _pool(tmp_path)
    bad = list(pool.items)
    bad[3] = AudioItem(uid=bad[3].uid, audio_path=bad[3].audio_path, pcm16_sha256="0" * 64, n_samples=bad[3].n_samples)
    ckpt = InferenceCheckpoint(tmp_path / "ckpt", binding=pool.binding, ordered_uids=pool.nb11.ordered_uids, shard_size=3)
    with pytest.raises(AudioIdentityError):
        run_asr_stage(ckpt, bad, fake_asr, batch_size=1)


class _FakeCtcModel:
    """Logits peak at round(frame_mean * 50); vocab 0=pad/blank, 1..3 letters, 4 special, 5 unk."""

    def __init__(self, hop=160, vocab=6):
        import torch

        self.hop = hop
        self.vocab = vocab
        self.param = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        yield self.param

    def eval(self):
        return self

    def __call__(self, input_values, attention_mask=None):
        import torch

        b, n = input_values.shape
        t = n // self.hop
        frames = input_values[:, : t * self.hop].reshape(b, t, self.hop).mean(-1)
        idx = torch.arange(self.vocab, dtype=torch.float32)
        logits = -((frames.unsqueeze(-1) * 50.0 - idx) ** 2)
        return SimpleNamespace(logits=logits)


class _FakeProcessor:
    tokenizer = SimpleNamespace(pad_token_id=0, unk_token_id=5, all_special_ids=[0, 4, 5])
    chars = {1: "a", 2: "b", 3: " ", 5: "?"}

    def __call__(self, wavs, sampling_rate, padding, return_tensors):
        import torch

        n = max(len(w) for w in wavs)
        arr = np.zeros((len(wavs), n), dtype=np.float32)
        for i, w in enumerate(wavs):
            arr[i, : len(w)] = w
        return SimpleNamespace(input_values=torch.tensor(arr))

    def batch_decode(self, ids):
        out = []
        for row in ids.tolist():
            prev, chars = None, []
            for t in row:
                if t != prev and t not in (0, 4):
                    chars.append(self.chars.get(t, ""))
                prev = t
            out.append("".join(chars))
        return out


def test_asr_adapter_matches_rq1_inference_text(tmp_path):
    from src.data_utils import normalize_bahnar_ctc_v1
    from src.rq1_inference import run_asr_predictions

    pool = _pool(tmp_path)
    model, proc = _FakeCtcModel(), _FakeProcessor()
    adapter = HfCtcAsrAdapter(model, proc)
    ours = [normalize_bahnar_ctc_v1(adapter([load_verified_waveform(it)])[0]["text_raw"]) for it in pool.items]
    frame = pd.DataFrame({"record_uid": [it.uid for it in pool.items]})
    rq1 = run_asr_predictions(model=model, processor=proc, frame=frame,
                              audio_paths={it.uid: it.audio_path for it in pool.items}, batch_size=1)
    assert ours == rq1["asr_pred_bahnar"].tolist()
    assert any(ours)
    conf = adapter([load_verified_waveform(pool.items[0])])[0]
    assert conf["asr_token_count"] > 0 and conf["asr_mean_logprob"] <= 0.0


def test_verified_waveform_equals_soundfile_float32(tmp_path):
    import soundfile as sf

    pool = _pool(tmp_path)
    ours = load_verified_waveform(pool.items[0])
    ref, sr = sf.read(str(pool.items[0].audio_path), dtype="float32")
    assert sr == 16000 and np.array_equal(ours, ref)


class _FakeMtTokenizer:
    pad_token_id, bos_token_id, eos_token_id, unk_token_id = 1, 0, 2, 3
    all_special_ids = [0, 1, 2, 3]

    def __init__(self):
        self.c2i, self.i2c = {}, {}

    def _id(self, ch):
        if ch not in self.c2i:
            self.c2i[ch] = 10 + len(self.c2i)
            self.i2c[self.c2i[ch]] = ch
        return self.c2i[ch]

    def __call__(self, texts, padding=False, truncation=False, max_length=None, return_tensors=None,
                 add_special_tokens=True, return_attention_mask=True):
        import torch

        single = isinstance(texts, str)
        seqs = [[0] + [self._id(c) for c in t] + [2] for t in ([texts] if single else texts)]
        if truncation and max_length:
            seqs = [s if len(s) <= max_length else s[: max_length - 1] + [2] for s in seqs]
        if return_tensors == "pt":
            n = max(len(s) for s in seqs)
            ids = torch.tensor([s + [1] * (n - len(s)) for s in seqs])
            return {"input_ids": ids, "attention_mask": (ids != 1).long()}
        return {"input_ids": seqs[0] if single else seqs}

    def batch_decode(self, ids, skip_special_tokens=True):
        rows = ids.tolist() if hasattr(ids, "tolist") else ids
        return ["".join(self.i2c.get(t, "") for t in row if t not in self.all_special_ids) for row in rows]


class _FakeMtModel:
    vocab = 120

    def __init__(self):
        import torch

        self.param = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        yield self.param

    def eval(self):
        return self

    def generate(self, input_ids, attention_mask=None, max_length=20, num_beams=1, **_):
        import torch

        rows = []
        for row in input_ids.tolist():
            content = [t for t in row if t >= 10][::-1][: max_length - 2]
            rows.append([2] + content + [2])
        n = max(len(r) for r in rows)
        return torch.tensor([r + [1] * (n - len(r)) for r in rows])

    def __call__(self, input_ids, decoder_input_ids, attention_mask=None):
        import torch

        idx = torch.arange(self.vocab, dtype=torch.float64)
        logits = torch.sin(idx * (decoder_input_ids.unsqueeze(-1).double() + 1.0) * 0.1) * 3.0
        return SimpleNamespace(logits=logits)


def test_mt_adapter_matches_rq1_inference_text():
    from src.mt_normalize import normalize_mt_text_v1
    from src.rq1_inference import run_mt_from_asr

    tok, model = _FakeMtTokenizer(), _FakeMtModel()
    sources = ["bah nar", "kon tum", "a b c d e f g h i j k l m n o p"]
    adapter = HfSeq2SeqMtAdapter(model, tok, max_source_length=12, generation_max_length=10, num_beams=1)
    ours = adapter(sources)
    frame = pd.DataFrame({"record_uid": ["u0", "u1", "u2"], "asr_pred_bahnar": sources})
    rq1 = run_mt_from_asr(model=model, tokenizer=tok, asr_predictions=frame, max_source_length=12,
                          generation_max_length=10, num_beams=1, batch_size=3)
    assert [normalize_mt_text_v1(o["text_raw"]) for o in ours] == rq1["c0_pred_vi"].tolist()
    assert ours[0]["mt_token_count"] == len("bah nar")
    assert ours[2]["source_truncated"] and ours[2]["hit_max_length"]
    assert not ours[0]["source_truncated"]
    assert all(o["mt_mean_logprob"] < 0 for o in ours)


def test_d0_adapter_matches_rq1_inference_text(tmp_path):
    import torch

    from src.mt_normalize import normalize_mt_text_v1
    from src.rq1_inference import run_direct_predictions

    pool = _pool(tmp_path)

    class Tok:
        lang_code_to_id = {"vi_VN": 7}

        def batch_decode(self, ids, skip_special_tokens=True):
            return [" ".join(f"t{t}" for t in row if t >= 10) for row in ids.tolist()]

    class Feat:
        def __call__(self, wavs, sampling_rate, padding, return_tensors, return_attention_mask):
            n = max(len(w) for w in wavs)
            arr = np.zeros((len(wavs), n), dtype=np.float32)
            for i, w in enumerate(wavs):
                arr[i, : len(w)] = w
            return {"input_values": torch.tensor(arr), "attention_mask": torch.ones(len(wavs), n, dtype=torch.long)}

    class Model:
        param = torch.nn.Parameter(torch.zeros(1))

        def parameters(self):
            yield self.param

        def eval(self):
            return self

        def generate(self, input_values, attention_mask=None, max_length=10, num_beams=1, forced_bos_token_id=None):
            assert forced_bos_token_id == 7
            return torch.tensor([[2, 7, 10 + int(input_values[i].abs().sum()) % 9, 2] for i in range(input_values.shape[0])])

    adapter = HfDirectD0Adapter(Model(), Feat(), Tok(), target_lang="vi_VN", generation_max_length=10, num_beams=1)
    ours = [normalize_mt_text_v1(adapter([load_verified_waveform(it)])[0]["text_raw"]) for it in pool.items]
    frame = pd.DataFrame({"record_uid": [it.uid for it in pool.items]})
    rq1 = run_direct_predictions(model=Model(), feature_extractor=Feat(), tokenizer=Tok(), frame=frame,
                                 audio_paths={it.uid: it.audio_path for it in pool.items}, target_lang="vi_VN",
                                 generation_max_length=10, num_beams=1, batch_size=1)
    assert ours == rq1["d0_pred_vi"].tolist()


def test_adapters_run_under_inference_mode():
    import torch

    seen = []

    class Spy(_FakeCtcModel):
        def __call__(self, input_values, attention_mask=None):
            seen.append(torch.is_inference_mode_enabled())
            return super().__call__(input_values, attention_mask)

    HfCtcAsrAdapter(Spy(), _FakeProcessor())([np.zeros(1600, dtype=np.float32)])
    assert seen == [True]


# --------------------------------------------------------------------------- #
# D. Confidence                                                                #
# --------------------------------------------------------------------------- #
def _lp(rows):
    return np.log(np.asarray(rows, dtype=np.float64))


def test_ctc_confidence_collapses_runs_and_drops_blank_and_special():
    lp = _lp([[0.1, 0.8, 0.1], [0.1, 0.6, 0.3], [0.9, 0.05, 0.05], [0.2, 0.1, 0.7]])
    ids = [1, 1, 0, 2]
    out = ctc_greedy_confidence(lp, ids, blank_id=0)
    assert out["asr_token_count"] == 2
    expected = np.mean([np.mean(np.log([0.8, 0.6])), np.log(0.7)])
    assert out["asr_mean_logprob"] == pytest.approx(expected)
    assert out["asr_confidence_raw"] == pytest.approx(math.exp(expected))
    assert out["asr_blank_frame_fraction"] == pytest.approx(0.25)
    special = ctc_greedy_confidence(lp, ids, blank_id=0, ignore_ids=[2])
    assert special["asr_token_count"] == 1


def test_ctc_confidence_is_length_normalised_and_empty_is_none():
    one = ctc_greedy_confidence(_lp([[0.2, 0.8]]), [1], blank_id=0)
    many = ctc_greedy_confidence(_lp([[0.2, 0.8], [0.9, 0.1]] * 5), [1, 0] * 5, blank_id=0)
    assert one["asr_mean_logprob"] == pytest.approx(many["asr_mean_logprob"])
    assert many["asr_token_count"] == 5
    empty = ctc_greedy_confidence(_lp([[0.9, 0.1], [0.9, 0.1]]), [0, 0], blank_id=0)
    assert empty["asr_token_count"] == 0 and empty["asr_mean_logprob"] is None and empty["asr_confidence_raw"] is None


def test_seq2seq_confidence_excludes_special_tokens():
    out = seq2seq_mean_logprob([-0.5, -1.0, -3.0, -9.0], [11, 12, 2, 1], ignore_ids=[0, 1, 2])
    assert out["mt_token_count"] == 2 and out["mt_mean_logprob"] == pytest.approx(-0.75)
    none = seq2seq_mean_logprob([-1.0], [2], ignore_ids=[2])
    assert none["mt_mean_logprob"] is None
    nonfinite = seq2seq_mean_logprob([float("-inf")], [11], ignore_ids=[])
    assert nonfinite["mt_mean_logprob"] is None


def test_text_validity():
    assert text_is_valid("xin chào") and text_is_valid("")
    assert not text_is_valid("bad\ufffd") and not text_is_valid("x\x00") and not text_is_valid(None)
    assert not text_is_valid("\ud800")


# --------------------------------------------------------------------------- #
# E. D0 agreement                                                              #
# --------------------------------------------------------------------------- #
def test_agreement_features():
    same = agreement_features("xin chào bạn", "xin chào bạn")
    assert same["teacher_d0_agreement_chrf"] == pytest.approx(100.0)
    a = agreement_features("xin chào", "chào bạn")["teacher_d0_agreement_chrf"]
    b = agreement_features("chào bạn", "xin chào")["teacher_d0_agreement_chrf"]
    assert a == pytest.approx(b)
    assert agreement_features("xin", "")["teacher_d0_agreement_chrf"] == 0.0
    assert agreement_features(None, "x")["teacher_d0_agreement_chrf"] is None


def test_d0_enabled_applies_to_every_record(tmp_path):
    pool = _pool(tmp_path, d0_enabled=True)
    _, _, records, released = _run(pool, tmp_path / "ckpt")
    assert released == ["asr", "mt", "d0"]
    assert all(r["d0_status"] == STATUS_OK for r in records)
    assert all(r["teacher_d0_agreement_chrf"] is not None for r in records)


def test_previous_model_is_collectable_before_the_next_loader(tmp_path):
    import gc
    import weakref

    pool = _pool(tmp_path, d0_enabled=True)
    refs = {}
    released = []

    class ResidentModel:
        def __init__(self, stage):
            self.stage = stage

    def make(stage, fn):
        def load():
            gc.collect()
            for prev, ref in refs.items():
                assert ref() is None, f"{prev} was still resident when {stage} started loading"
            model = ResidentModel(stage)
            refs[stage] = weakref.ref(model)

            def release(stage=stage):
                released.append(stage)

            return {"adapter": fn, "resident": model, "release": release}

        return load

    ckpt = InferenceCheckpoint(tmp_path / "ckpt", binding=pool.binding, ordered_uids=pool.nb11.ordered_uids, shard_size=3)
    run_peak_safe_teacher_inference(
        ckpt=ckpt, items=pool.items, decoding=pool.decoding,
        load_asr=make("asr", fake_asr), load_mt=make("mt", fake_mt), load_d0=make("d0", fake_d0),
    )
    assert released == ["asr", "mt", "d0"]
    gc.collect()
    assert all(ref() is None for ref in refs.values())


def test_release_runs_and_model_dies_when_stage_raises(tmp_path):
    import gc
    import weakref

    pool = _pool(tmp_path)
    released = []
    refbox = {}

    class ResidentModel:
        pass

    def load_asr():
        model = ResidentModel()
        refbox["ref"] = weakref.ref(model)

        def adapter(waves):
            raise RuntimeError("bad runtime")

        def release():
            released.append("asr")

        return {"adapter": adapter, "resident": model, "release": release}

    def load_mt():
        raise AssertionError("MT loader started while the ASR failure was unresolved")

    ckpt = InferenceCheckpoint(tmp_path / "ckpt", binding=pool.binding, ordered_uids=pool.nb11.ordered_uids, shard_size=3)
    with pytest.raises(RuntimeError, match="bad runtime"):
        run_peak_safe_teacher_inference(
            ckpt=ckpt, items=pool.items, decoding=pool.decoding, load_asr=load_asr, load_mt=load_mt,
        )
    assert released == ["asr"]
    gc.collect()
    assert refbox["ref"]() is None


def test_d0_loader_must_match_contract(tmp_path):
    pool = _pool(tmp_path)
    ckpt = InferenceCheckpoint(tmp_path / "ckpt", binding=pool.binding, ordered_uids=pool.nb11.ordered_uids, shard_size=3)
    loader = lambda: {"adapter": fake_d0}  # noqa: E731
    with pytest.raises(RuntimeError, match="disabled"):
        run_peak_safe_teacher_inference(ckpt=ckpt, items=pool.items, decoding=pool.decoding,
                                        load_asr=loader, load_mt=loader, load_d0=loader)
    pool_on = _pool(tmp_path / "on", d0_enabled=True)
    ckpt_on = InferenceCheckpoint(tmp_path / "ckpt_on", binding=pool_on.binding, ordered_uids=pool_on.nb11.ordered_uids)
    with pytest.raises(RuntimeError, match="no D0 loader"):
        run_peak_safe_teacher_inference(ckpt=ckpt_on, items=pool_on.items, decoding=pool_on.decoding,
                                        load_asr=loader, load_mt=loader)


# --------------------------------------------------------------------------- #
# H. Resume                                                                    #
# --------------------------------------------------------------------------- #
def test_resume_after_interruption_equals_uninterrupted(tmp_path):
    pool = _pool(tmp_path)
    _, _, clean, _ = _run(pool, tmp_path / "clean")
    calls = {"n": 0}

    def dies_after_one_shard(waves):
        calls["n"] += 1
        if calls["n"] > 3:
            raise RuntimeError("out of memory (simulated interruption)")
        return fake_asr(waves)

    with pytest.raises(RuntimeError):
        _run(pool, tmp_path / "resumed", asr=dies_after_one_shard)
    ckpt, report, resumed, _ = _run(pool, tmp_path / "resumed")
    assert resumed == clean
    assert report["stages"][0]["reused_shards"] == 1


def test_corrupt_shard_is_recomputed_alone(tmp_path):
    pool = _pool(tmp_path)
    _, _, clean, _ = _run(pool, tmp_path / "ckpt")
    shard = tmp_path / "ckpt" / "asr" / "shard-00001.jsonl"
    shard.write_text(shard.read_text().replace("bah", "BAH", 1))
    counted = []

    def counting(waves):
        counted.extend(len(w) for w in waves)
        return fake_asr(waves)

    ckpt, report, again, _ = _run(pool, tmp_path / "ckpt", asr=counting)
    assert again == clean
    assert len(counted) == 3
    assert any(q.startswith("asr/1:") for q in ckpt.quarantined)
    assert (tmp_path / "ckpt" / "quarantine" / "asr").is_dir()
    mt = next(s for s in report["stages"] if s["stage"] == "mt")
    assert mt["computed_shards"] == 0


def test_stale_mt_dependency_is_recomputed(tmp_path):
    pool = _pool(tmp_path)
    _run(pool, tmp_path / "ckpt")
    ckpt = InferenceCheckpoint(tmp_path / "ckpt", binding=pool.binding, ordered_uids=pool.nb11.ordered_uids, shard_size=3)
    rows = ckpt.load_shard("asr", 0)
    rows[0]["asr_text_norm"] = "changed"
    ckpt.write_shard("asr", 0, rows)
    res = run_mt_stage(ckpt, fake_mt, batch_size=8)
    assert res["computed_shards"] == 1 and res["reused_shards"] == ckpt.n_shards - 1


@pytest.mark.parametrize("field,value,reason", [
    ("stage", "not-asr", "bad_stage"),
    ("shard_index", 99, "bad_shard_index"),
    ("uids_sha256", "0" * 64, "bad_uids_sha256"),
    ("rows_sha256", "0" * 64, "hash_mismatch"),
    ("n_rows", 0, "bad_n_rows"),
])
def test_corrupt_shard_metadata_recomputes_only_that_shard(tmp_path, field, value, reason):
    pool = _pool(tmp_path)
    _, _, clean, _ = _run(pool, tmp_path / "ckpt")
    meta_path = tmp_path / "ckpt" / "asr" / "shard-00001.json"
    meta = json.loads(meta_path.read_text())
    meta[field] = value
    meta_path.write_text(json.dumps(meta))
    counted = []

    def counting(waves):
        counted.extend(waves)
        return fake_asr(waves)

    ckpt, _, again, _ = _run(pool, tmp_path / "ckpt", asr=counting)
    assert again == clean
    assert len(counted) == 3
    assert any(q.startswith(f"asr/1:{reason}") for q in ckpt.quarantined)


def test_corrupt_dependency_metadata_recomputes_only_that_shard(tmp_path):
    pool = _pool(tmp_path)
    _run(pool, tmp_path / "ckpt")
    meta_path = tmp_path / "ckpt" / "mt" / "shard-00000.json"
    meta = json.loads(meta_path.read_text())
    meta["depends_on"] = {"asr": "0" * 64}
    meta_path.write_text(json.dumps(meta))
    ckpt = InferenceCheckpoint(tmp_path / "ckpt", binding=pool.binding, ordered_uids=pool.nb11.ordered_uids, shard_size=3)
    res = run_mt_stage(ckpt, fake_mt, batch_size=8)
    assert res["computed_shards"] == 1 and res["reused_shards"] == ckpt.n_shards - 1
    assert any(q.startswith("mt/0:stale_dependency") for q in ckpt.quarantined)


def test_shard_binding_metadata_mismatch_fails_closed(tmp_path):
    pool = _pool(tmp_path)
    _run(pool, tmp_path / "ckpt")
    meta_path = tmp_path / "ckpt" / "asr" / "shard-00000.json"
    data_path = meta_path.with_suffix(".jsonl")
    meta = json.loads(meta_path.read_text())
    meta["binding_sha256"] = "0" * 64
    meta_path.write_text(json.dumps(meta))
    ckpt = InferenceCheckpoint(tmp_path / "ckpt", binding=pool.binding, ordered_uids=pool.nb11.ordered_uids, shard_size=3)
    with pytest.raises(CheckpointBindingError, match="different binding"):
        ckpt.load_shard("asr", 0)
    assert meta_path.is_file() and data_path.is_file()
    assert not (tmp_path / "ckpt" / "quarantine").exists()


def test_binding_mismatch_fails_closed_and_archive_never_deletes(tmp_path):
    pool = _pool(tmp_path)
    _run(pool, tmp_path / "ckpt")
    other = {**pool.binding, "teacher_contract_sha256": "0" * 64}
    with pytest.raises(CheckpointBindingError):
        InferenceCheckpoint(tmp_path / "ckpt", binding=other, ordered_uids=pool.nb11.ordered_uids, shard_size=3)
    with pytest.raises(CheckpointBindingError):
        InferenceCheckpoint(tmp_path / "ckpt", binding=pool.binding, ordered_uids=pool.nb11.ordered_uids, shard_size=4)
    moved = archive_checkpoint(tmp_path / "ckpt")
    assert moved.is_dir() and (moved / "state.json").is_file() and not (tmp_path / "ckpt").exists()


def test_shard_membership_is_enforced(tmp_path):
    pool = _pool(tmp_path)
    ckpt = InferenceCheckpoint(tmp_path / "ckpt", binding=pool.binding, ordered_uids=pool.nb11.ordered_uids, shard_size=3)
    with pytest.raises(RuntimeError, match="fixed membership"):
        ckpt.write_shard("asr", 0, [{"uid": "seg-0002"}, {"uid": "seg-0001"}, {"uid": "seg-0000"}])


# --------------------------------------------------------------------------- #
# J. Anti-leakage + G_test guard                                               #
# --------------------------------------------------------------------------- #
def test_g_test_paths_and_splits_are_refused(tmp_path):
    for name in ("rq1_test.csv", "rq1_test.parquet", "frozen_test_audio", "g_test"):
        with pytest.raises(GTestAccessError):
            assert_not_g_test_path(tmp_path / name)
    assert_not_g_test_path(tmp_path / "test_resolving_test0" / "rq1_validation.csv")
    ledger = DataAccessLedger()
    with pytest.raises(GTestAccessError):
        ledger.record("g_test", tmp_path / "x.csv", "nope")
    with pytest.raises(GTestAccessError):
        ledger.record("g_validation", tmp_path / "rq1_test.csv", "nope")
    ledger.record("g_validation", tmp_path / "rq1_validation.csv", "calibration")
    assert not ledger.g_test_accessed and ledger.as_dict()["entries"][0]["file"] == "rq1_validation.csv"


def test_nb12_output_dir_cannot_be_inside_protected_trees(tmp_path):
    with pytest.raises(RuntimeError, match="protected"):
        assert_nb12_output_dir(tmp_path / "artifacts" / "rq2" / "u_clean" / "x", tmp_path)
    with pytest.raises(RuntimeError, match="protected"):
        assert_nb12_output_dir(tmp_path / "data" / "manifests" / "out", tmp_path)
    assert assert_nb12_output_dir(tmp_path / "artifacts" / "rq2" / "pseudo_labels", tmp_path)


def test_reference_fields_never_enter_pseudo_path(tmp_path):
    frame = pd.DataFrame({"record_uid": ["v1"], "audio_path": ["x.wav"], "sha256_pcm": ["a" * 64],
                          "n_samples": [16000], "text_vi": ["gold"]})
    with pytest.raises(ReferenceLeakError):
        items_from_audio_frame(frame)
    pool = _pool(tmp_path)
    _, _, records, _ = _run(pool, tmp_path / "ckpt")
    assert not (set(records[0]) & FORBIDDEN_REFERENCE_FIELDS)
    ckpt = InferenceCheckpoint(tmp_path / "ckpt2", binding=pool.binding, ordered_uids=pool.nb11.ordered_uids, shard_size=3)
    with pytest.raises(ReferenceLeakError):
        ckpt.write_shard("asr", 0, [{"uid": u, "text_vi": "gold"} for u in ckpt.shard_uids(0)])


def test_nb12_sources_never_name_frozen_test_files_or_train():
    for path in NB12_SOURCES:
        text = path.read_text(encoding="utf-8")
        assert "rq1_test.csv" not in text and "rq1_test.parquet" not in text
        assert ".train(" not in text and "Trainer(" not in text and "do_sample=True" not in text
        assert "optimizer" not in text.lower()


# --------------------------------------------------------------------------- #
# Notebook static checks (never executed)                                      #
# --------------------------------------------------------------------------- #
def _notebook_code():
    nb = json.loads(NB12_NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_notebook_flags_default_off_and_cells_compile():
    cells = _notebook_code()
    joined = "\n".join(cells)
    assert "RUN_FULL_PIPELINE = False" in joined
    assert "QUALITY_SCORE_FROZEN = False" in joined
    for name in ("NB11_INPUT_READY", "TEACHER_CONTRACT_READY", "QUALITY_SCORE_FROZEN", "RUN_FULL_PIPELINE"):
        assert f'print("{name}' in joined or f"print('{name}" in joined or f'"{name}"' in joined
    for i, src in enumerate(cells):
        compile(src, f"nb12_cell_{i}", "exec")


def test_notebook_has_no_selection_training_or_g_test():
    joined = "\n".join(_notebook_code())
    for banned in ("rq1_test.csv", "rq1_test.parquet", ".train(", "Trainer(", ".sample(", "nlargest(", "do_sample=True"):
        assert banned not in joined, banned
    assert "raise" in joined and "NB11_INPUT_READY" in joined


# --------------------------------------------------------------------------- #
# G_validation audio cache                                                     #
# --------------------------------------------------------------------------- #
_GOLD = "HUMAN_VI_REFERENCE_MUST_NOT_ENTER_CACHE"
_MANIFEST = "f" * 64
_DATASET = "org/bahnar-validation"
_REVISION = "a" * 40
_PARQUET = "validation/shard-00000.parquet"


def _wav_bytes(fill: int, n_samples: int = 1600) -> bytes:
    from src.asr_full_pcm import build_wav_bytes

    return build_wav_bytes(bytes([fill & 0xFF, 0]) * n_samples, 16000)


class _ValidationReader:
    def __init__(self, audio_by_index):
        self.audio_by_index = {int(k): v for k, v in audio_by_index.items()}
        self.calls = []

    def __call__(self, *, cache_dir, downloaded):
        def read(ref, needed):
            self.calls.append(sorted(int(i) for i in needed))
            for idx in needed:
                record_id, wav = self.audio_by_index[int(idx)]
                yield int(idx), {
                    "id": record_id,
                    "audio": {"bytes": wav},
                    "text_vi": _GOLD,
                    "reference_vi": _GOLD,
                }
        return read


def _validation_frame(order=("va", "vb")):
    rows = {
        "va": {"record_uid": "va", "record_id": "rec-a", "shard_row_index": 0},
        "vb": {"record_uid": "vb", "record_id": "rec-b", "shard_row_index": 1},
    }
    return pd.DataFrame([
        {**rows[uid], "parquet_file": _PARQUET, "split": "validation", "text_vi": _GOLD, "gold_vi": _GOLD}
        for uid in order
    ])


def _materialize(audio_dir, reader, frame=None, manifest=_MANIFEST):
    return materialize_validation_audio(
        _validation_frame() if frame is None else frame,
        dataset_id=_DATASET,
        parquet_revision=_REVISION,
        manifest_sha256=manifest,
        audio_dir=audio_dir,
        parquet_cache_dir=audio_dir.parent / "_parquet_cache",
        reader_factory=reader,
    )


def _cache_blob(audio_dir: Path) -> str:
    parts = []
    for path in sorted(audio_dir.rglob("*")):
        if path.is_file() and "quarantine" not in path.parts:
            parts.append(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(parts)


def test_validation_audio_cache_builds_then_reuses_without_rereading_source(tmp_path):
    reader = _ValidationReader({0: ("rec-a", _wav_bytes(1)), 1: ("rec-b", _wav_bytes(2, 3200))})
    audio = tmp_path / "validation_audio"
    first = _materialize(audio, reader)
    assert first["n_validation_total"] == 2 and first["n_rebuilt"] == 2 and first["n_reused"] == 0
    assert (audio / "validation_audio_cache_index.jsonl").is_file()
    assert (audio / "validation_audio_cache_state.json").is_file()
    assert len(list(audio.glob("uid_*.wav"))) == 2 and len(list(audio.glob("uid_*.json"))) == 2
    assert _GOLD not in _cache_blob(audio)
    assert "text_vi" not in first["audio_frame"].columns
    identity = first["validation_audio_identity_sha256"]
    cache_id = first["cache_identity_sha256"]

    second = _materialize(audio, reader)
    assert second["n_reused"] == 2 and second["n_rebuilt"] == 0 and second["n_invalidated"] == 0
    assert reader.calls == [[0, 1]]
    assert second["validation_audio_identity_sha256"] == identity
    assert second["cache_identity_sha256"] == cache_id
    assert second["ordered_uid_hash"] == first["ordered_uid_hash"]


def test_one_corrupt_validation_wav_rebuilds_only_that_uid(tmp_path):
    reader = _ValidationReader({0: ("rec-a", _wav_bytes(1)), 1: ("rec-b", _wav_bytes(2, 3200))})
    audio = tmp_path / "validation_audio"
    first = _materialize(audio, reader)
    bad = Path(first["audio_frame"].loc[first["audio_frame"]["record_uid"] == "vb", "audio_path"].iloc[0])
    good = Path(first["audio_frame"].loc[first["audio_frame"]["record_uid"] == "va", "audio_path"].iloc[0])
    good_bytes = good.read_bytes()
    bad.write_bytes(b"not-a-wav")
    again = _materialize(audio, reader)
    assert again["n_invalidated"] == 1 and again["n_rebuilt"] == 1 and again["n_reused"] == 1
    assert reader.calls == [[0, 1], [1]]
    assert good.read_bytes() == good_bytes
    assert again["validation_audio_identity_sha256"] == first["validation_audio_identity_sha256"]


def test_changed_source_binding_invalidates_only_the_affected_uid(tmp_path):
    reader = _ValidationReader({
        0: ("rec-a", _wav_bytes(1)),
        1: ("rec-b", _wav_bytes(2, 3200)),
        7: ("rec-b", _wav_bytes(9, 2400)),
    })
    audio = tmp_path / "validation_audio"
    first = _materialize(audio, reader)
    frame = _validation_frame()
    frame.loc[frame["record_uid"] == "vb", "shard_row_index"] = 7
    again = _materialize(audio, reader, frame=frame)
    assert again["n_invalidated"] == 1 and again["n_rebuilt"] == 1 and again["n_reused"] == 1
    assert reader.calls[-1] == [7]
    assert again["validation_audio_identity_sha256"] != first["validation_audio_identity_sha256"]
    kept = again["audio_frame"].loc[again["audio_frame"]["record_uid"] == "va"].iloc[0]
    original = first["audio_frame"].loc[first["audio_frame"]["record_uid"] == "va"].iloc[0]
    assert kept["sha256_pcm"] == original["sha256_pcm"]


def test_bad_sidecar_rebuilds_only_that_uid(tmp_path):
    reader = _ValidationReader({0: ("rec-a", _wav_bytes(1)), 1: ("rec-b", _wav_bytes(2))})
    audio = tmp_path / "validation_audio"
    first = _materialize(audio, reader)
    wav = Path(first["audio_frame"].loc[first["audio_frame"]["record_uid"] == "vb", "audio_path"].iloc[0])
    wav.with_suffix(".json").write_text("{", encoding="utf-8")
    again = _materialize(audio, reader)
    assert again["n_invalidated"] == 1 and again["n_rebuilt"] == 1 and again["n_reused"] == 1
    assert reader.calls[-1] == [1]
    assert _GOLD not in _cache_blob(audio)


def test_partial_cache_files_are_never_reused(tmp_path):
    reader = _ValidationReader({0: ("rec-a", _wav_bytes(1)), 1: ("rec-b", _wav_bytes(2))})
    audio = tmp_path / "validation_audio"
    first = _materialize(audio, reader)
    paths = {row.record_uid: Path(row.audio_path) for row in first["audio_frame"].itertuples()}
    payload = paths["va"].read_bytes()
    paths["va"].with_name(paths["va"].name + ".tmp").write_bytes(payload)
    paths["va"].unlink()
    paths["va"].with_suffix(".json").unlink()
    paths["vb"].write_bytes(payload[:8])
    paths["vb"].with_suffix(".json").unlink()
    again = _materialize(audio, reader)
    assert again["n_reused"] == 0 and again["n_rebuilt"] == 2
    assert reader.calls[-1] == [0, 1]
    assert paths["va"].with_name(paths["va"].name + ".tmp").is_file()
    assert paths["va"].is_file() and paths["va"].stat().st_size > 8


def test_reordered_validation_input_keeps_per_uid_audio_and_changes_order_hash(tmp_path):
    reader = _ValidationReader({0: ("rec-a", _wav_bytes(1)), 1: ("rec-b", _wav_bytes(2, 3200))})
    audio = tmp_path / "validation_audio"
    first = _materialize(audio, reader)
    again = _materialize(audio, reader, frame=_validation_frame(("vb", "va")))
    assert again["n_reused"] == 2 and again["n_rebuilt"] == 0
    assert reader.calls == [[0, 1]]
    assert again["ordered_uid_hash"] != first["ordered_uid_hash"]
    assert again["validation_audio_identity_sha256"] != first["validation_audio_identity_sha256"]
    for uid in ("va", "vb"):
        before = first["audio_frame"].loc[first["audio_frame"]["record_uid"] == uid, "sha256_pcm"].iloc[0]
        after = again["audio_frame"].loc[again["audio_frame"]["record_uid"] == uid, "sha256_pcm"].iloc[0]
        assert before == after


def test_validation_audio_identity_is_deterministic_and_path_independent(tmp_path):
    def once(root):
        reader = _ValidationReader({0: ("rec-a", _wav_bytes(1)), 1: ("rec-b", _wav_bytes(2, 3200))})
        return _materialize(root / "validation_audio", reader)

    left = once(tmp_path / "left")
    right = once(tmp_path / "right")
    assert left["validation_audio_identity_sha256"] == right["validation_audio_identity_sha256"]
    assert left["cache_identity_sha256"] == right["cache_identity_sha256"]
    assert left["ordered_uid_hash"] == right["ordered_uid_hash"]
    assert left["audio_frame"]["audio_path"].iloc[0] != right["audio_frame"]["audio_path"].iloc[0]
    for root in (tmp_path / "left" / "validation_audio", tmp_path / "right" / "validation_audio"):
        blob = _cache_blob(root)
        assert str(root) not in blob and "/Users/" not in blob and _GOLD not in blob
        assert left["validation_audio_identity_sha256"] not in str(root)


def test_changed_validation_pcm_changes_identity_and_rejects_old_checkpoint(tmp_path):
    audio = tmp_path / "validation_audio"
    first = _materialize(audio, _ValidationReader({0: ("rec-a", _wav_bytes(1)), 1: ("rec-b", _wav_bytes(2))}))
    teacher, d0, *_ = fake_contracts()
    uids = first["audio_frame"]["record_uid"].astype(str).tolist()
    binding = inference_binding(
        pool="g_validation",
        input_identity_sha256=first["validation_audio_identity_sha256"],
        teacher_contract_sha256=teacher["teacher_contract_sha256"],
        d0_agreement_contract_sha256=d0["d0_agreement_contract_sha256"],
        decoding_config_sha256=teacher["decoding_config_sha256"],
    )
    InferenceCheckpoint(tmp_path / "ckpt", binding=binding, ordered_uids=uids, shard_size=8)
    for path in first["audio_frame"]["audio_path"]:
        Path(path).write_bytes(b"broken")
    second = _materialize(audio, _ValidationReader({0: ("rec-a", _wav_bytes(3)), 1: ("rec-b", _wav_bytes(4, 2400))}))
    assert second["n_rebuilt"] == 2
    assert second["validation_audio_identity_sha256"] != first["validation_audio_identity_sha256"]
    rebound = inference_binding(
        pool="g_validation",
        input_identity_sha256=second["validation_audio_identity_sha256"],
        teacher_contract_sha256=teacher["teacher_contract_sha256"],
        d0_agreement_contract_sha256=d0["d0_agreement_contract_sha256"],
        decoding_config_sha256=teacher["decoding_config_sha256"],
    )
    with pytest.raises(CheckpointBindingError):
        InferenceCheckpoint(tmp_path / "ckpt", binding=rebound, ordered_uids=uids, shard_size=8)


def test_validation_materialisation_never_uses_g_test(tmp_path):
    reader = _ValidationReader({0: ("rec-a", _wav_bytes(1)), 1: ("rec-b", _wav_bytes(2))})
    with pytest.raises(GTestAccessError):
        _materialize(tmp_path / "g_test" / "audio", reader)
    leaked = _validation_frame()
    leaked["split"] = "test"
    with pytest.raises(RuntimeError, match="G_validation"):
        _materialize(tmp_path / "validation_audio", reader, frame=leaked)
    assert reader.calls == []
