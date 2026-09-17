"""Offline unit tests for data_utils (Notebook 02 helpers)."""

from __future__ import annotations

import hashlib
import io
import ipaddress
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.audio_utils import check_waveform, waveform_to_mono_float32
from src.data_utils import (
    AUDIO_PROCESSING_VERSION,
    CONTAMINATION_CHAR_SCHEMA,
    CONTAMINATION_ROW_SCHEMA,
    FROZEN_TEST_ALLOWED_COLUMNS,
    FROZEN_TEST_FORBIDDEN_COLUMNS,
    NOTEBOOK03_COMPAT_MAX_DURATION,
    NOTEBOOK03_COMPAT_MIN_DURATION,
    ResponseTooLargeError,
    assert_no_signed_urls_persisted,
    asset_headers,
    audit_train_script_contamination,
    build_cache_provenance,
    build_char_ctc_vocab,
    cache_provenance_matches,
    check_manifest_contract,
    classify_character,
    compute_contamination_summary,
    compute_ordered_uid_hash,
    compute_uid_set_hash,
    download_audio_bytes,
    download_audio_bytes_with_redirects,
    duration_bin,
    encode_text_with_vocab,
    export_clean_split_contract,
    extract_audio_src,
    filter_candidates_by_duration,
    find_oov_characters,
    find_oov_rows,
    generate_run_id,
    init_audio_result_schema,
    is_missing_scalar,
    is_valid_sha256,
    load_cache_sidecar,
    load_frozen_test_restricted,
    normalize_bahnar_ctc_v1,
    parse_bool,
    parse_bool_series,
    process_audio_candidate,
    redact_url,
    request_with_retries,
    resample_audio,
    run_metadata,
    safe_cache_filename,
    select_representative_samples,
    set_dns_resolver_for_testing,
    sha256_file,
    source_qa_to_result_fields,
    strip_signed_url_columns,
    texts_match_after_light_norm,
    to_mono_float32,
    uniform_window_offsets,
    validate_cache_wav,
    validate_https_url,
    verify_clean_split_contract,
    verify_json_metadata,
    verify_report_metadata,
    viewer_headers,
    write_cache_sidecar,
    write_dataframe_csv,
    write_pcm16_wav,
    read_wav,
)


def test_sha256_file(tmp_path: Path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello-bahnar")
    assert sha256_file(p) == hashlib.sha256(b"hello-bahnar").hexdigest()
    assert is_valid_sha256(sha256_file(p))
    assert not is_valid_sha256("abc")
    assert not is_valid_sha256("5a7b13c87d0be684b351dfb02ea97f2efa64b1e7bc2a7e46ba2dd206ca9253f")


def test_parse_bool_robust():
    assert parse_bool(True) is True
    assert parse_bool(False) is False
    assert parse_bool(1) is True
    assert parse_bool(0) is False
    assert parse_bool("true") is True
    assert parse_bool("FALSE") is False
    assert parse_bool("1") is True
    assert parse_bool("0") is False
    with pytest.raises(ValueError):
        parse_bool("yes")
    s = pd.Series(["False", "True", "0", "1"])
    out = parse_bool_series(s)
    assert out.tolist() == [False, True, False, True]


def test_missing_scalar_nulls():
    assert is_missing_scalar(None)
    assert is_missing_scalar(np.nan)
    assert is_missing_scalar(pd.NA)
    assert normalize_bahnar_ctc_v1(None) == ""
    assert normalize_bahnar_ctc_v1(np.nan) == ""
    assert normalize_bahnar_ctc_v1(pd.NA) == ""
    assert normalize_bahnar_ctc_v1("") == ""
    assert not is_missing_scalar("NA")
    assert normalize_bahnar_ctc_v1("NA") == "na"


def test_normalize_keeps_bahnar_marks():
    out = normalize_bahnar_ctc_v1("Pơlei Bahnar — TEST")
    assert "ơ" in out
    assert "—" not in out
    assert "'" in normalize_bahnar_ctc_v1("don\u2019t")


def test_vocab_and_oov():
    texts = ["Pơlei Bahnar", "pơlei  bahnar!", "Kon"]
    a = build_char_ctc_vocab(texts)
    b = build_char_ctc_vocab(list(reversed(texts)))
    assert a == b
    assert a["[PAD]"] == 0
    assert " " not in a
    oov = find_oov_characters(["abz"], build_char_ctc_vocab(["abc"]))
    assert "z" in set(oov["character"])
    df = pd.DataFrame([{"record_uid": "u1", "text_bahnar": "abz"}])
    rows = find_oov_rows(df, build_char_ctc_vocab(["abc"]))
    assert len(rows) == 1
    assert "z" in rows.iloc[0]["oov_characters"]


def test_uniform_window_offsets_head_tail_and_small():
    offs = uniform_window_offsets(113_830, 10, 80, seed=42)
    assert offs[0] == 0
    assert offs[-1] == 113_830 - 10
    assert offs == sorted(offs)
    assert len(offs) == len(set(offs))
    assert uniform_window_offsets(113_830, 10, 80, 42) == offs
    assert uniform_window_offsets(5, 10, 80, 42) == [0]
    assert uniform_window_offsets(0, 10, 80, 42) == []
    offs2 = uniform_window_offsets(100, 10, 2, 42)
    assert offs2 == [0, 90]


# ============================================================================
# Contamination classifier tests - CRITICAL
# ============================================================================

class TestClassifyCharacter:
    """Tests for classify_character() - check script detection BEFORE combining_mark."""
    
    def test_khmer_vowel_signs_are_unexpected_script(self):
        """Khmer vowel signs (category M*) must be flagged as unexpected_script."""
        # KHMER VOWEL SIGN AA (U+17B6) - category Mc
        assert classify_character("ា") == "unexpected_script"
        # KHMER VOWEL SIGN I (U+17B7) - category Mn
        assert classify_character("ិ") == "unexpected_script"
        # KHMER SIGN COENG (U+17D2) - category Mn
        assert classify_character("្") == "unexpected_script"
    
    def test_thai_combining_marks_are_unexpected_script(self):
        """Thai combining marks must be flagged as unexpected_script."""
        # THAI CHARACTER MAI HAN-AKAT (U+0E31) - category Mn
        assert classify_character("ั") == "unexpected_script"
        # THAI CHARACTER SARA I (U+0E34) - category Mn
        assert classify_character("ิ") == "unexpected_script"
    
    def test_khmer_letters_are_unexpected_script(self):
        """Khmer letters must be flagged as unexpected_script."""
        assert classify_character("ក") == "unexpected_script"
        assert classify_character("ខ") == "unexpected_script"
    
    def test_thai_letters_are_unexpected_script(self):
        """Thai letters must be flagged as unexpected_script."""
        assert classify_character("ก") == "unexpected_script"
        assert classify_character("ข") == "unexpected_script"
    
    def test_cyrillic_is_unexpected_script(self):
        """Cyrillic letters must be flagged as unexpected_script."""
        # Cyrillic Small Letter A (U+0430) - looks like Latin 'a'
        assert classify_character("а") == "unexpected_script"
    
    def test_latin_combining_marks_are_accepted(self):
        """Latin combining marks should be combining_mark, not unexpected."""
        # COMBINING MACRON BELOW (U+0331)
        assert classify_character("\u0331") == "combining_mark"
        # COMBINING ACUTE ACCENT (U+0301)
        assert classify_character("\u0301") == "combining_mark"
    
    def test_latin_letters_accepted(self):
        """Valid Latin letters should be latin_letter."""
        assert classify_character("ơ") == "latin_letter"
        assert classify_character("ư") == "latin_letter"
        assert classify_character("ă") == "latin_letter"
    
    def test_ipa_extensions_accepted(self):
        """IPA extensions used in Bahnar should be latin_letter."""
        # LATIN SMALL LETTER B WITH HOOK (U+0253)
        assert classify_character("ɓ") == "latin_letter"
        # LATIN SMALL LETTER I WITH STROKE (U+0268)
        assert classify_character("ɨ") == "latin_letter"
        # LATIN SMALL LETTER SCHWA (U+0259)
        assert classify_character("ə") == "latin_letter"
    
    def test_control_characters(self):
        """Control characters should be control_character."""
        assert classify_character("\x00") == "control_character"
        assert classify_character("\x1f") == "control_character"
    
    def test_digits(self):
        """Digits should be digit."""
        assert classify_character("0") == "digit"
        assert classify_character("9") == "digit"
    
    def test_allowed_punctuation(self):
        """Allowed punctuation should be allowed_punctuation."""
        assert classify_character("'") == "allowed_punctuation"
        assert classify_character("-") == "allowed_punctuation"


class TestContaminationAudit:
    """Tests for audit_train_script_contamination()."""
    
    def test_detects_khmer_in_vocab(self):
        """Audit should detect Khmer characters including combining marks."""
        vocab = {"[PAD]": 0, "[UNK]": 1, "|": 2, "a": 3, "ក": 4, "ា": 5}
        char_df, row_df = audit_train_script_contamination(vocab)
        
        # Should flag both Khmer letter and vowel sign
        flagged = set(char_df["character"])
        assert "ក" in flagged
        assert "ា" in flagged
        assert "a" not in flagged
    
    def test_char_report_columns(self):
        """Character report should have required columns."""
        vocab = {"[PAD]": 0, "[UNK]": 1, "|": 2, "ក": 3}
        char_df, _ = audit_train_script_contamination(vocab)
        
        required = ["character", "codepoint", "unicode_name", "category",
                    "vocab_id", "classification", "flags", 
                    "occurrence_count", "affected_row_count"]
        for col in required:
            assert col in char_df.columns, f"Missing column: {col}"
    
    def test_row_report_columns(self):
        """Row report should have required columns."""
        vocab = {"[PAD]": 0, "[UNK]": 1, "|": 2, "a": 3, "ក": 4}
        train_df = pd.DataFrame([{
            "record_uid": "u1",
            "record_id": "r1",
            "source_label": "s1",
            "group_id": "g1",
            "recording_group_id": "rg1",
            "text_bahnar": "aកb"
        }])
        _, row_df = audit_train_script_contamination(vocab, train_df=train_df)
        
        required = ["record_uid", "record_id", "source_label", "group_id",
                    "recording_group_id", "text_bahnar", "text_bahnar_norm",
                    "flagged_characters", "flagged_codepoints", 
                    "flagged_unicode_names", "flags"]
        for col in required:
            assert col in row_df.columns, f"Missing column: {col}"
    
    def test_summary_only_counts_unexpected_script(self):
        """Summary should only count rows with unexpected_script, not other flags."""
        vocab = {"[PAD]": 0, "[UNK]": 1, "|": 2, "a": 3, "ក": 4}  # Khmer
        train_df = pd.DataFrame([
            {"record_uid": "u1", "group_id": "g1", "text_bahnar": "aកb"},  # Has Khmer
        ])
        char_df, row_df = audit_train_script_contamination(vocab, train_df=train_df)
        summary = compute_contamination_summary(char_df, row_df)
        
        assert summary["unexpected_script_characters"] == 1
        assert summary["affected_rows"] == 1


# ============================================================================
# Representative sampling tests
# ============================================================================

def test_filter_candidates_by_duration_keeps_inclusive_bounds():
    """Only keep numeric finite durations in [0.5, 30.0]."""
    df = pd.DataFrame({
        "record_uid": [f"u{i}" for i in range(8)],
        "group_id": [f"g{i}" for i in range(8)],
        "duration_seconds": [0.49, 0.5, 1.0, 30.0, 30.01, float("nan"), -1.0, float("inf")],
    })
    out = filter_candidates_by_duration(df)
    assert set(out["record_uid"]) == {"u1", "u2", "u3"}
    durs = pd.to_numeric(out["duration_seconds"])
    assert (durs >= NOTEBOOK03_COMPAT_MIN_DURATION).all()
    assert (durs <= NOTEBOOK03_COMPAT_MAX_DURATION).all()


def test_filter_candidates_by_duration_empty_and_missing_col():
    empty = pd.DataFrame(columns=["record_uid", "duration_seconds"])
    assert len(filter_candidates_by_duration(empty)) == 0
    with pytest.raises(ValueError, match="duration_seconds"):
        filter_candidates_by_duration(pd.DataFrame({"record_uid": ["a"]}))


def test_duration_filter_then_sample_hits_validation_target_70():
    """After duration filter, sampling can still hit validation target=70."""
    rows = []
    for i in range(120):
        # Mix of in-range and out-of-range durations
        dur = 1.0 + (i % 40) * 0.5  # 1.0 .. 20.5 for most
        if i % 10 == 0:
            dur = 45.0  # too long — must be dropped before sampling
        rows.append({
            "record_uid": f"v{i}",
            "group_id": f"g{i % 20}",
            "source_label": f"s{i % 4}",
            "duration_seconds": dur,
            "final_split": "validation",
        })
    raw = pd.DataFrame(rows)
    filtered = filter_candidates_by_duration(raw)
    assert (pd.to_numeric(filtered["duration_seconds"]) <= 30.0).all()
    assert (pd.to_numeric(filtered["duration_seconds"]) >= 0.5).all()
    assert filtered["record_uid"].nunique() >= 70

    sel = select_representative_samples(filtered, n=70, seed=42)
    assert len(sel) == 70
    assert sel["record_uid"].nunique() == 70
    assert set(sel["record_uid"]).issubset(set(filtered["record_uid"]))
    # No out-of-range records selected
    assert set(sel["record_uid"]).isdisjoint(set(raw.loc[raw["duration_seconds"] > 30, "record_uid"]))


def test_duration_filter_then_sample_deterministic_and_clean_only():
    """Deterministic selection; never picks UIDs outside the provided clean candidate pool."""
    clean_uids = {f"clean-{i}" for i in range(100)}
    rows = []
    for i in range(100):
        rows.append({
            "record_uid": f"clean-{i}",
            "group_id": f"g{i % 10}",
            "source_label": f"s{i % 3}",
            "duration_seconds": 2.0 + (i % 20),
        })
    # Contaminated / out-of-split decoys with valid duration must not appear if not in candidates
    decoys = pd.DataFrame([
        {"record_uid": "outside-1", "group_id": "gx", "source_label": "sx", "duration_seconds": 5.0},
        {"record_uid": "outside-2", "group_id": "gy", "source_label": "sy", "duration_seconds": 8.0},
    ])
    pool = filter_candidates_by_duration(pd.DataFrame(rows))
    assert set(pool["record_uid"]).issubset(clean_uids)

    sel1 = select_representative_samples(pool, n=40, seed=7)
    sel2 = select_representative_samples(pool, n=40, seed=7)
    sel3 = select_representative_samples(pool, n=40, seed=99)
    assert list(sel1["record_uid"]) == list(sel2["record_uid"])
    assert set(sel1["record_uid"]) != set(sel3["record_uid"])
    assert set(sel1["record_uid"]).issubset(clean_uids)
    assert set(sel1["record_uid"]).isdisjoint(set(decoys["record_uid"]))


def test_representative_sampling_balanced_data():
    """Test with balanced data."""
    rows = []
    for src in ["a", "b", "c"]:
        for g in range(8):
            for k in range(4):
                rows.append({
                    "record_uid": f"{src}-{g}-{k}",
                    "group_id": f"g-{src}-{g}",
                    "source_label": src,
                    "duration_seconds": 2 + (g % 5) * 4,
                })
    df = pd.DataFrame(rows)
    sel = select_representative_samples(df, n=24, seed=42, max_per_group=5)
    assert len(sel) == 24
    assert sel["record_uid"].nunique() == 24
    assert sel["source_label"].nunique() >= 2
    assert sel["group_id"].nunique() >= 8


def test_representative_sampling_imbalanced_must_return_n():
    """Relaxed pass must re-examine rejected records and return exactly n."""
    rows = []
    for k in range(100):
        rows.append({
            "record_uid": f"big-{k}",
            "group_id": "g-big",
            "source_label": "big_source",
            "duration_seconds": 10.0,
        })
    for src in ["small1", "small2"]:
        for k in range(5):
            rows.append({
                "record_uid": f"{src}-{k}",
                "group_id": f"g-{src}",
                "source_label": src,
                "duration_seconds": 5.0,
            })
    
    df = pd.DataFrame(rows)
    unique = df["record_uid"].nunique()
    
    sel = select_representative_samples(df, n=20, seed=42)
    assert len(sel) == 20
    assert sel["record_uid"].nunique() == 20
    
    sel2 = select_representative_samples(df, n=200, seed=42)
    assert len(sel2) == unique


def test_representative_sampling_relaxed_pass_reexamines():
    """Relaxed pass revisits records rejected in strict pass."""
    rows = []
    for k in range(50):
        rows.append({
            "record_uid": f"same-group-{k}",
            "group_id": "single-group",
            "source_label": "single-source",
            "duration_seconds": 10.0,
        })
    
    df = pd.DataFrame(rows)
    sel = select_representative_samples(df, n=20, seed=42, max_per_group=2)
    assert len(sel) == 20


def test_representative_sampling_no_duplicates():
    """No duplicate record_uid in selection."""
    rows = [{"record_uid": f"u{i}", "group_id": "g1", "source_label": "s1", "duration_seconds": 5.0} 
            for i in range(100)]
    df = pd.DataFrame(rows)
    sel = select_representative_samples(df, n=50, seed=42)
    assert sel["record_uid"].nunique() == len(sel)


def test_representative_sampling_deterministic():
    """Same seed must produce same result."""
    rows = [{"record_uid": f"u{i}", "group_id": f"g{i%5}", "source_label": f"s{i%3}", 
             "duration_seconds": float(i)} for i in range(100)]
    df = pd.DataFrame(rows)
    sel1 = select_representative_samples(df, n=30, seed=123)
    sel2 = select_representative_samples(df, n=30, seed=123)
    assert list(sel1["record_uid"]) == list(sel2["record_uid"])


# ============================================================================
# URL and HTTP tests
# ============================================================================

def test_extract_and_url_validation():
    payload = {"audio": [{"src": "https://cdn.example/a.wav?X-Amz-Signature=abc"}]}
    url = extract_audio_src(payload)
    assert url.startswith("https://")
    # Use reject_private=False for basic validation test (DNS won't resolve test domains)
    validate_https_url(url, reject_private=False)
    with pytest.raises(ValueError):
        validate_https_url("http://insecure.example/a.wav", reject_private=False)
    with pytest.raises(ValueError):
        validate_https_url("https://user:pass@host/a.wav", reject_private=False)
    assert "X-Amz" not in redact_url(url)


def test_asset_headers_have_no_authorization():
    assert "Authorization" not in asset_headers()
    vh = viewer_headers("secret-token")
    assert vh["Authorization"] == "Bearer secret-token"


def test_download_audio_does_not_forward_authorization():
    sleeps = []
    captured = {}

    class FakeResp:
        status_code = 200
        headers = {}
        _content = b"WAVDATA"

        def raise_for_status(self):
            return None
        
        def iter_content(self, chunk_size=None):
            return iter([self._content])
        
        def close(self):
            pass

    class FakeSession:
        def request(self, method, url, headers=None, params=None, timeout=None, allow_redirects=None, stream=None):
            captured["headers"] = dict(headers or {})
            return FakeResp()

    # Use reject_private=False since cdn.example won't resolve
    data = download_audio_bytes(
        "https://cdn.example/file.wav",
        session=FakeSession(),
        sleep_fn=lambda s: sleeps.append(s),
        reject_private=False,
    )
    assert data == b"WAVDATA"
    assert "Authorization" not in captured["headers"]


def test_private_localhost_url_rejected():
    """Private/localhost URLs must be rejected."""
    with pytest.raises(ValueError, match="Private|localhost|reserved"):
        validate_https_url("https://localhost/file.wav", reject_private=True)
    with pytest.raises(ValueError, match="Private|localhost|reserved"):
        validate_https_url("https://127.0.0.1/file.wav", reject_private=True)


def test_redirect_to_private_ip_blocked():
    """Redirect to private IP must be blocked."""
    call_count = {"n": 0}
    
    class FakeResp:
        def __init__(self, status, location=None):
            self.status_code = status
            self.headers = {"Location": location} if location else {}
            self._content = b"data"
        
        def raise_for_status(self):
            pass
        
        def iter_content(self, chunk_size=None):
            return iter([self._content])
        
        def close(self):
            pass
    
    class FakeSession:
        def request(self, method, url, headers=None, timeout=None, allow_redirects=None, stream=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return FakeResp(302, "https://127.0.0.1/private")
            return FakeResp(200)
    
    # First URL passes (reject_private=True but we mock DNS to return public IP)
    # Redirect to 127.0.0.1 should fail
    def mock_resolver(host):
        if host == "cdn.example":
            return [ipaddress.ip_address("8.8.8.8")]
        return []
    
    with pytest.raises(RuntimeError, match="Invalid redirect|Private"):
        download_audio_bytes_with_redirects(
            "https://cdn.example/file.wav",
            session=FakeSession(),
            reject_private=True,
            dns_resolver=mock_resolver,
        )


def test_redirect_to_http_blocked():
    """Redirect to HTTP must be blocked."""
    class FakeResp:
        status_code = 302
        headers = {"Location": "http://insecure.example/file"}
        _content = b""
        def raise_for_status(self):
            pass
        def iter_content(self, chunk_size=None):
            return iter([self._content])
        def close(self):
            pass
    
    class FakeSession:
        def request(self, *args, **kwargs):
            return FakeResp()
    
    def mock_resolver(host):
        return [ipaddress.ip_address("8.8.8.8")]
    
    with pytest.raises(RuntimeError, match="Invalid redirect|HTTPS"):
        download_audio_bytes_with_redirects(
            "https://cdn.example/file.wav",
            session=FakeSession(),
            reject_private=True,
            dns_resolver=mock_resolver,
        )


def test_valid_redirect_works():
    """Valid HTTPS redirect should work."""
    call_count = {"n": 0}
    
    class FakeResp:
        def __init__(self, status, location=None, content=b""):
            self.status_code = status
            self.headers = {"Location": location} if location else {}
            self._content = content
        
        def raise_for_status(self):
            pass
        
        def iter_content(self, chunk_size=None):
            return iter([self._content])
        
        def close(self):
            pass
    
    class FakeSession:
        def request(self, method, url, headers=None, timeout=None, allow_redirects=None, stream=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return FakeResp(302, "https://cdn2.example/actual.wav")
            return FakeResp(200, content=b"AUDIODATA")
    
    def mock_resolver(host):
        return [ipaddress.ip_address("8.8.8.8")]
    
    data = download_audio_bytes_with_redirects(
        "https://cdn.example/file.wav",
        session=FakeSession(),
        reject_private=True,
        dns_resolver=mock_resolver,
    )
    assert data == b"AUDIODATA"
    assert call_count["n"] == 2


def test_redirect_loop_limited():
    """Redirect loop must be limited."""
    class FakeResp:
        status_code = 302
        headers = {"Location": "https://cdn.example/loop"}
        _content = b""
        def raise_for_status(self):
            pass
        def iter_content(self, chunk_size=None):
            return iter([self._content])
        def close(self):
            pass
    
    class FakeSession:
        def request(self, *args, **kwargs):
            return FakeResp()
    
    def mock_resolver(host):
        return [ipaddress.ip_address("8.8.8.8")]
    
    with pytest.raises(RuntimeError, match="Too many redirects"):
        download_audio_bytes_with_redirects(
            "https://cdn.example/file.wav",
            session=FakeSession(),
            max_redirects=5,
            reject_private=True,
            dns_resolver=mock_resolver,
        )


def test_http_404_no_retry():
    """HTTP 404 should not be retried."""
    import requests as real_requests
    call_count = {"n": 0}
    
    class FakeResp:
        status_code = 404
        headers = {}
        content = b""
        
        def raise_for_status(self):
            err = real_requests.HTTPError("404 Client Error")
            err.response = self
            raise err
    
    class FakeSession:
        def request(self, *args, **kwargs):
            call_count["n"] += 1
            return FakeResp()
    
    with pytest.raises(Exception):
        request_with_retries(
            "GET",
            "https://example.com/notfound",
            headers={},
            session=FakeSession(),
            sleep_fn=lambda s: None,
            max_attempts=6,
        )
    # Should only call once, no retries for 404
    assert call_count["n"] == 1


def test_retry_http_429_respects_retry_after():
    sleeps = []
    calls = {"n": 0}

    class FakeResp:
        def __init__(self, status, retry_after=None):
            self.status_code = status
            self.headers = {"Retry-After": retry_after} if retry_after is not None else {}
            self.content = b"{}"

        def raise_for_status(self):
            if self.status_code >= 400:
                import requests
                err = requests.HTTPError(f"{self.status_code}")
                err.response = self
                raise err

        def json(self):
            return {"ok": True}

    class FakeSession:
        def request(self, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] < 3:
                return FakeResp(429, retry_after="0.01")
            return FakeResp(200)

    resp = request_with_retries(
        "GET",
        "https://datasets-server.huggingface.co/rows",
        headers=viewer_headers(),
        session=FakeSession(),
        sleep_fn=lambda s: sleeps.append(s),
        max_attempts=6,
        rng=__import__("random").Random(0),
    )
    assert calls["n"] == 3
    assert sleeps
    assert all(s <= 30 for s in sleeps)


# ============================================================================
# Audio processing tests
# ============================================================================

def test_int16_pcm_canonical_stereo_and_layouts():
    mono = np.zeros(1000, dtype=np.int16)
    mono[100:200] = 10000
    out = to_mono_float32(mono)
    assert out.dtype == np.float32
    assert 0.2 < float(np.max(np.abs(out))) < 0.4
    
    st = np.zeros((1000, 2), dtype=np.int16)
    st[100:200, :] = 10000
    out2 = to_mono_float32(st)
    assert out2.ndim == 1
    assert 0.2 < float(np.max(np.abs(out2))) < 0.4
    
    cf = np.zeros((2, 1000), dtype=np.int16)
    cf[:, 100:200] = 10000
    out3 = to_mono_float32(cf)
    assert out3.ndim == 1
    
    with pytest.raises(ValueError):
        waveform_to_mono_float32(np.array([]))


def test_resample_and_cache_sha(tmp_path: Path):
    sr = 8000
    t = np.linspace(0, 1, sr, endpoint=False)
    wav = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    out = resample_audio(wav, sr, 16_000)
    assert abs(len(out) - 16_000) <= 2
    path = tmp_path / "x.wav"
    write_pcm16_wav(path, out, 16_000)
    data, sr2 = read_wav(path)
    assert sr2 == 16000 and data.ndim == 1
    digest = sha256_file(path)
    assert is_valid_sha256(digest)


# ============================================================================
# Cache provenance tests
# ============================================================================

def test_cache_provenance_fingerprint():
    source_qa = {"ok": True, "hard_ok": True, "reason": "ok", "quality_warnings": [], 
                 "duration_sec": 1.0, "rms": 0.1, "clip_ratio": 0.0, "sampling_rate": 16000}
    prov = build_cache_provenance(
        record_uid="u1",
        dataset_revision="a" * 40,
        target_sr=16000,
        source_qa=source_qa,
        cache_sha256="b" * 64,
    )
    assert prov["audio_processing_version"] == AUDIO_PROCESSING_VERSION
    assert cache_provenance_matches(
        prov, record_uid="u1", dataset_revision="a" * 40, target_sr=16000, cache_sha256="b" * 64
    )
    assert not cache_provenance_matches(
        prov, record_uid="u1", dataset_revision="a" * 40, target_sr=16000, cache_sha256="c" * 64
    )


def test_cache_provenance_corrupt_sidecar_returns_false():
    """Corrupt sidecar must return False, not crash."""
    assert cache_provenance_matches(None, record_uid="u", dataset_revision="a"*40, 
                                    target_sr=16000, cache_sha256="b"*64) is False
    assert cache_provenance_matches("not a dict", record_uid="u", dataset_revision="a"*40,
                                    target_sr=16000, cache_sha256="b"*64) is False
    assert cache_provenance_matches({}, record_uid="u", dataset_revision="a"*40,
                                    target_sr=16000, cache_sha256="b"*64) is False


def test_validate_cache_wav_and_sidecar(tmp_path: Path):
    wav = np.zeros(1600, dtype=np.float32)
    wav[100:200] = 0.2
    path = tmp_path / "c.wav"
    write_pcm16_wav(path, wav, 16000)
    v = validate_cache_wav(path, target_sr=16000, recompute_sha=True)
    assert v["cache_hard_ok"]
    assert v["cache_channels"] == 1
    assert is_valid_sha256(v["cache_sha256"])
    
    source_qa = {"ok": True, "hard_ok": True, "reason": "ok", "quality_warnings": [],
                 "duration_sec": 0.1, "rms": 0.01, "clip_ratio": 0.0, "sampling_rate": 8000}
    prov = build_cache_provenance(
        record_uid="u1", dataset_revision="a"*40, target_sr=16000,
        source_qa=source_qa, cache_sha256=v["cache_sha256"],
    )
    write_cache_sidecar(path, prov)
    loaded = load_cache_sidecar(path)
    assert cache_provenance_matches(
        loaded, record_uid="u1", dataset_revision="a"*40, 
        target_sr=16000, cache_sha256=v["cache_sha256"]
    )


# ============================================================================
# Frozen test seal tests  
# ============================================================================

def test_frozen_test_restricted_load(tmp_path: Path):
    """Frozen test loader only loads allowed columns."""
    (tmp_path / "src").mkdir()
    (tmp_path / "requirements.txt").write_text("x\n")
    man = tmp_path / "data" / "manifests"
    man.mkdir(parents=True)
    
    test_csv = man / "rq1_test.csv"
    test_csv.write_text(
        "record_uid,record_id,source_split,group_id,text_bahnar,text_vi,audio_path\n"
        "u1,r1,test,g1,secret_text,secret_vi,/audio/path\n"
    )
    
    df, report = load_frozen_test_restricted(tmp_path)
    
    assert "text_bahnar" not in df.columns
    assert "text_vi" not in df.columns
    assert "audio_path" not in df.columns
    assert "record_uid" in df.columns
    assert "group_id" in df.columns
    assert report["passed"] is True


def test_check_manifest_contract_uses_restricted_test_loader(tmp_path: Path):
    """check_manifest_contract uses restricted loader for test."""
    (tmp_path / "src").mkdir()
    (tmp_path / "requirements.txt").write_text("x\n")
    man = tmp_path / "data" / "manifests"
    aud = tmp_path / "data" / "audit"
    man.mkdir(parents=True)
    aud.mkdir(parents=True)
    
    test_content = "record_uid,record_id,source_split,group_id,pair_key,split,text_bahnar,text_vi\n"
    for i in range(215):
        src = "validation" if i < 205 else "test"
        test_content += f"u{i},r{i},{src},g{i},pk{i},test,secret{i},vi{i}\n"
    (man / "rq1_test.csv").write_text(test_content)
    
    test_sha = hashlib.sha256(test_content.encode()).hexdigest()
    summary = {
        "dataset_id": "test",
        "expected_dataset_revision": "a"*40,
        "dataset_commit_sha": "a"*40,
        "manifest_sha256": {"rq1_test.csv": test_sha},
    }
    (man / "split_summary.json").write_text(json.dumps(summary))
    
    checks = check_manifest_contract(
        root=tmp_path,
        dataset_id="test",
        expected_revision="a"*40,
        expected_counts={"test": 215},
        hub_sha="a"*40,
        split_summary=summary,
    )
    
    restricted_check = checks.loc[checks["check"] == "test_restricted_load"]
    assert len(restricted_check) == 1
    assert bool(restricted_check.iloc[0]["passed"]) is True


# ============================================================================
# JSON metadata verification tests
# ============================================================================

class TestVerifyJsonMetadata:
    """Tests for verify_json_metadata()."""
    
    def test_correct_metadata_returns_true(self, tmp_path: Path):
        """JSON with correct metadata returns True."""
        path = tmp_path / "report.json"
        path.write_text(json.dumps({
            "run_id": "run-123",
            "dataset_revision": "rev-456",
            "processing_version": "v1",
        }))
        
        assert verify_json_metadata(
            path,
            expected_run_id="run-123",
            expected_revision="rev-456",
            expected_processing_version="v1",
        ) is True
    
    def test_wrong_run_id_returns_false(self, tmp_path: Path):
        """Wrong run_id returns False."""
        path = tmp_path / "report.json"
        path.write_text(json.dumps({
            "run_id": "old-run",
            "dataset_revision": "rev-456",
            "processing_version": "v1",
        }))
        
        assert verify_json_metadata(
            path,
            expected_run_id="new-run",
            expected_revision="rev-456",
            expected_processing_version="v1",
        ) is False
    
    def test_wrong_revision_returns_false(self, tmp_path: Path):
        """Wrong dataset_revision returns False."""
        path = tmp_path / "report.json"
        path.write_text(json.dumps({
            "run_id": "run-123",
            "dataset_revision": "wrong-rev",
            "processing_version": "v1",
        }))
        
        assert verify_json_metadata(
            path,
            expected_run_id="run-123",
            expected_revision="correct-rev",
            expected_processing_version="v1",
        ) is False
    
    def test_wrong_processing_version_returns_false(self, tmp_path: Path):
        """Wrong processing_version returns False."""
        path = tmp_path / "report.json"
        path.write_text(json.dumps({
            "run_id": "run-123",
            "dataset_revision": "rev-456",
            "processing_version": "v1",
        }))
        
        assert verify_json_metadata(
            path,
            expected_run_id="run-123",
            expected_revision="rev-456",
            expected_processing_version="v2",
        ) is False
    
    def test_invalid_json_returns_false(self, tmp_path: Path):
        """Invalid JSON returns False."""
        path = tmp_path / "report.json"
        path.write_text("not valid json {{{")
        
        assert verify_json_metadata(
            path,
            expected_run_id="run-123",
            expected_revision="rev-456",
            expected_processing_version="v1",
        ) is False
    
    def test_missing_file_returns_false(self, tmp_path: Path):
        """Missing file returns False."""
        path = tmp_path / "nonexistent.json"
        
        assert verify_json_metadata(
            path,
            expected_run_id="run-123",
            expected_revision="rev-456",
            expected_processing_version="v1",
        ) is False


# ============================================================================
# Tokenizer clean save tests
# ============================================================================

def test_save_tokenizer_clean_removes_stale_tokens(tmp_path: Path):
    """save_tokenizer_clean should remove stale BOS/EOS tokens."""
    try:
        from transformers import Wav2Vec2CTCTokenizer
    except ImportError:
        pytest.skip("transformers not installed")
    
    from src.data_utils import save_tokenizer_clean
    
    out_dir = tmp_path / "tokenizer"
    out_dir.mkdir()
    
    # Create stale added_tokens.json
    stale_added = {"<s>": 210, "</s>": 211}
    (out_dir / "added_tokens.json").write_text(json.dumps(stale_added))
    
    vocab = {"[PAD]": 0, "[UNK]": 1, "|": 2, "a": 3, "b": 4}
    
    verification = save_tokenizer_clean(vocab, out_dir, target_sampling_rate=16000)
    
    assert verification["passed"] is True
    assert verification["bos_token"] is None
    assert verification["eos_token"] is None
    assert verification["has_stale_bos_eos"] is False
    
    # Reload and verify
    reloaded = Wav2Vec2CTCTokenizer.from_pretrained(str(out_dir))
    
    assert len(reloaded) == len(vocab)
    assert reloaded.bos_token is None
    assert reloaded.eos_token is None
    assert "<s>" not in reloaded.get_vocab()
    assert "</s>" not in reloaded.get_vocab()
    
    # Check added_tokens.json if it exists
    added_path = out_dir / "added_tokens.json"
    if added_path.exists():
        added = json.loads(added_path.read_text())
        assert "<s>" not in added
        assert "</s>" not in added


# ============================================================================
# Audio candidate processing tests
# ============================================================================

class TestProcessAudioCandidate:
    """Tests for process_audio_candidate helper."""
    
    def test_fresh_path_creates_cache(self, tmp_path: Path):
        """Fresh path: no cache → download → create cache."""
        import soundfile as sf
        from src.data_utils import set_dns_resolver_for_testing
        
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        
        # Fake audio data
        sr = 16000
        t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
        audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
        
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV")
        audio_bytes = buf.getvalue()
        
        download_called = {"n": 0}
        def fake_download(url):
            download_called["n"] += 1
            return audio_bytes
        
        def fake_decode(data):
            return sf.read(io.BytesIO(data), always_2d=False)
        
        # Mock DNS to return public IP
        set_dns_resolver_for_testing(lambda h: [ipaddress.ip_address("8.8.8.8")])
        try:
            result = process_audio_candidate(
                record_uid="test-uid",
                record_id="r1",
                final_split="train",
                source_label="s1",
                group_id="g1",
                duration_seconds_meta=0.5,
                text_identity_ok=True,
                audio_src="https://example.com/audio.wav",
                row_idx=0,
                cache_dir=cache_dir,
                project_root=tmp_path,
                dataset_revision="rev123",
                target_sr=16000,
                min_duration=0.05,
                max_duration=120.0,
                download_fn=fake_download,
                decode_fn=fake_decode,
            )
        finally:
            set_dns_resolver_for_testing(None)
        
        assert download_called["n"] == 1
        assert result["reused_cache"] is False
        assert result["source_qa_available"] is True
        assert result["source_hard_ok"] is True
        assert result["cache_hard_ok"] is True
        assert result["qa_hard_ok"] is True
    
    def test_reuse_path_no_download(self, tmp_path: Path):
        """Reuse path: valid cache → no download."""
        import soundfile as sf
        from src.data_utils import set_dns_resolver_for_testing
        
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        
        sr = 16000
        t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
        audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
        
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV")
        audio_bytes = buf.getvalue()
        
        def fake_download(url):
            return audio_bytes
        
        def fake_decode(data):
            return sf.read(io.BytesIO(data), always_2d=False)
        
        set_dns_resolver_for_testing(lambda h: [ipaddress.ip_address("8.8.8.8")])
        try:
            # First run creates cache
            result1 = process_audio_candidate(
                record_uid="test-uid",
                record_id="r1",
                final_split="train",
                source_label="s1",
                group_id="g1",
                duration_seconds_meta=0.5,
                text_identity_ok=True,
                audio_src="https://example.com/audio.wav",
                row_idx=0,
                cache_dir=cache_dir,
                project_root=tmp_path,
                dataset_revision="rev123",
                target_sr=16000,
                min_duration=0.05,
                max_duration=120.0,
                download_fn=fake_download,
                decode_fn=fake_decode,
            )
            assert result1["reused_cache"] is False
            
            # Second run reuses cache
            download_called = {"n": 0}
            def fake_download2(url):
                download_called["n"] += 1
                return audio_bytes
            
            result2 = process_audio_candidate(
                record_uid="test-uid",
                record_id="r1",
                final_split="train",
                source_label="s1",
                group_id="g1",
                duration_seconds_meta=0.5,
                text_identity_ok=True,
                audio_src="https://example.com/audio.wav",
                row_idx=0,
                cache_dir=cache_dir,
                project_root=tmp_path,
                dataset_revision="rev123",
                target_sr=16000,
                min_duration=0.05,
                max_duration=120.0,
                download_fn=fake_download2,
                decode_fn=fake_decode,
            )
        finally:
            set_dns_resolver_for_testing(None)
        
        assert download_called["n"] == 0  # No download
        assert result2["reused_cache"] is True
        assert result2["source_qa_available"] is True
        assert result2["qa_hard_ok"] is True
    
    def test_invalid_provenance_redownloads(self, tmp_path: Path):
        """Invalid provenance → must redownload."""
        import soundfile as sf
        from src.data_utils import set_dns_resolver_for_testing
        
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        
        sr = 16000
        t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
        audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
        
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV")
        audio_bytes = buf.getvalue()
        
        def fake_download(url):
            return audio_bytes
        
        def fake_decode(data):
            return sf.read(io.BytesIO(data), always_2d=False)
        
        set_dns_resolver_for_testing(lambda h: [ipaddress.ip_address("8.8.8.8")])
        try:
            # First run with revision "rev123"
            result1 = process_audio_candidate(
                record_uid="test-uid",
                record_id="r1",
                final_split="train",
                source_label="s1",
                group_id="g1",
                duration_seconds_meta=0.5,
                text_identity_ok=True,
                audio_src="https://example.com/audio.wav",
                row_idx=0,
                cache_dir=cache_dir,
                project_root=tmp_path,
                dataset_revision="rev123",
                target_sr=16000,
                min_duration=0.05,
                max_duration=120.0,
                download_fn=fake_download,
                decode_fn=fake_decode,
            )
            assert result1["reused_cache"] is False
            
            # Second run with DIFFERENT revision
            download_called = {"n": 0}
            def fake_download2(url):
                download_called["n"] += 1
                return audio_bytes
            
            result2 = process_audio_candidate(
                record_uid="test-uid",
                record_id="r1",
                final_split="train",
                source_label="s1",
                group_id="g1",
                duration_seconds_meta=0.5,
                text_identity_ok=True,
                audio_src="https://example.com/audio.wav",
                row_idx=0,
                cache_dir=cache_dir,
                project_root=tmp_path,
                dataset_revision="DIFFERENT_REVISION",  # Different!
                target_sr=16000,
                min_duration=0.05,
                max_duration=120.0,
                download_fn=fake_download2,
                decode_fn=fake_decode,
            )
        finally:
            set_dns_resolver_for_testing(None)
        
        # Must have downloaded because provenance doesn't match
        assert download_called["n"] == 1
        assert result2["reused_cache"] is False
    
    def test_fresh_and_reuse_same_schema(self, tmp_path: Path):
        """Fresh and reuse paths must return same schema."""
        import soundfile as sf
        from src.data_utils import set_dns_resolver_for_testing
        
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        
        sr = 16000
        t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
        audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
        
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV")
        audio_bytes = buf.getvalue()
        
        def fake_download(url):
            return audio_bytes
        
        def fake_decode(data):
            return sf.read(io.BytesIO(data), always_2d=False)
        
        set_dns_resolver_for_testing(lambda h: [ipaddress.ip_address("8.8.8.8")])
        try:
            # Fresh
            result_fresh = process_audio_candidate(
                record_uid="uid1",
                record_id="r1",
                final_split="train",
                source_label="s1",
                group_id="g1",
                duration_seconds_meta=0.5,
                text_identity_ok=True,
                audio_src="https://example.com/audio.wav",
                row_idx=0,
                cache_dir=cache_dir,
                project_root=tmp_path,
                dataset_revision="rev123",
                target_sr=16000,
                min_duration=0.05,
                max_duration=120.0,
                download_fn=fake_download,
                decode_fn=fake_decode,
            )
            
            # Reuse
            result_reuse = process_audio_candidate(
                record_uid="uid1",
                record_id="r1",
                final_split="train",
                source_label="s1",
                group_id="g1",
                duration_seconds_meta=0.5,
                text_identity_ok=True,
                audio_src="https://example.com/audio.wav",
                row_idx=0,
                cache_dir=cache_dir,
                project_root=tmp_path,
                dataset_revision="rev123",
                target_sr=16000,
                min_duration=0.05,
                max_duration=120.0,
                download_fn=fake_download,
                decode_fn=fake_decode,
            )
        finally:
            set_dns_resolver_for_testing(None)
        
        # Same keys
        assert set(result_fresh.keys()) == set(result_reuse.keys())


# ============================================================================
# Other tests
# ============================================================================

def test_no_persist_signed_url():
    df = pd.DataFrame({"record_uid": ["a"], "_audio_src": ["https://x?X-Amz-Signature=1"]})
    with pytest.raises(RuntimeError):
        assert_no_signed_urls_persisted(df)
    cleaned = strip_signed_url_columns(df)
    assert_no_signed_urls_persisted(cleaned)


def test_redact_strips_query():
    u = "https://cdn.example/a.wav?X-Amz-Signature=deadbeef&X-Amz-Credential=x"
    r = redact_url(u)
    assert "deadbeef" not in r
    assert "X-Amz" not in r
    assert r == "https://cdn.example/a.wav"


def test_run_metadata_includes_run_id():
    """Run metadata must include run_id."""
    run_id = generate_run_id()
    meta = run_metadata(
        run_id=run_id,
        dataset_id="test",
        dataset_revision="a"*40,
        seed=42,
        processing_version="v1",
    )
    assert meta["run_id"] == run_id
    assert "timestamp_utc" in meta


def test_verify_report_metadata_detects_stale(tmp_path: Path):
    """Stale reports must be detected."""
    csv_path = tmp_path / "report.csv"
    meta_path = tmp_path / "report.csv.meta.json"
    
    csv_path.write_text("col\nval\n")
    meta_path.write_text(json.dumps({
        "run_id": "old-run-id",
        "dataset_revision": "a"*40,
    }))
    
    assert verify_report_metadata(csv_path, expected_run_id="new-run-id", 
                                  expected_revision="a"*40) is False
    assert verify_report_metadata(csv_path, expected_run_id="old-run-id",
                                  expected_revision="a"*40) is True


def test_manifest_contract_sha_fail(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "requirements.txt").write_text("pandas\n")
    man = tmp_path / "data" / "manifests"
    aud = tmp_path / "data" / "audit"
    man.mkdir(parents=True)
    aud.mkdir(parents=True)
    summary = {
        "dataset_id": "cuong06/Bahnar_Vietnamese",
        "expected_dataset_revision": "a" * 40,
        "dataset_commit_sha": "a" * 40,
        "manifest_sha256": {
            "rq1_train.csv": "b" * 64,
            "rq1_validation.csv": "c" * 64,
            "rq1_test.csv": "d" * 64,
        },
    }
    (man / "split_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (man / "rq1_train.csv").write_text("record_uid\nx\n", encoding="utf-8")
    checks = check_manifest_contract(
        root=tmp_path,
        dataset_id="cuong06/Bahnar_Vietnamese",
        expected_revision="a" * 40,
        expected_counts={"train": 1, "validation": 1, "test": 215},
        hub_sha="a" * 40,
        split_summary=summary,
    )
    assert not bool(checks.loc[checks["check"] == "rq1_train.csv_sha_match", "passed"].iloc[0])


def test_texts_match_and_duration_bin():
    assert texts_match_after_light_norm("A  B", "A B")
    assert duration_bin(10.0) == "8-15s"
    assert safe_cache_filename("uid") == safe_cache_filename("uid")


def test_load_manifest_prefers_csv(tmp_path: Path):
    from src.data_utils import load_manifest

    man = tmp_path / "data" / "manifests"
    man.mkdir(parents=True)
    (tmp_path / "requirements.txt").write_text("x\n")
    (tmp_path / "src").mkdir()
    (man / "rq1_train.csv").write_text("record_uid,text_bahnar\na,hello\n", encoding="utf-8")
    (man / "rq1_train.parquet").write_text("not-real", encoding="utf-8")
    df = load_manifest("train", tmp_path)
    assert list(df["record_uid"]) == ["a"]


# ============================================================================
# Cache stats per split tests
# ============================================================================

class TestCacheStatsPerSplit:
    """Tests for compute_cache_stats_per_split and verify_cache_stats_consistency."""
    
    def test_independent_split_counts(self):
        """fresh/reuse counts must be independent per split."""
        from src.data_utils import compute_cache_stats_per_split
        
        qa_df = pd.DataFrame([
            {"final_split": "train", "reused_cache": True},
            {"final_split": "train", "reused_cache": True},
            {"final_split": "train", "reused_cache": False},  # 1 fresh
            {"final_split": "validation", "reused_cache": False},  # 1 fresh
            {"final_split": "validation", "reused_cache": True},  # 1 reused
        ])
        
        stats = compute_cache_stats_per_split(qa_df)
        
        assert stats["train"]["fresh_count"] == 1
        assert stats["train"]["reuse_count"] == 2
        assert stats["train"]["total"] == 3
        
        assert stats["validation"]["fresh_count"] == 1
        assert stats["validation"]["reuse_count"] == 1
        assert stats["validation"]["total"] == 2
    
    def test_all_fresh(self):
        """Test case where all caches are fresh."""
        from src.data_utils import compute_cache_stats_per_split
        
        qa_df = pd.DataFrame([
            {"final_split": "train", "reused_cache": False},
            {"final_split": "train", "reused_cache": False},
            {"final_split": "validation", "reused_cache": False},
        ])
        
        stats = compute_cache_stats_per_split(qa_df)
        
        assert stats["train"]["fresh_count"] == 2
        assert stats["train"]["reuse_count"] == 0
        assert stats["validation"]["fresh_count"] == 1
        assert stats["validation"]["reuse_count"] == 0
    
    def test_all_reused(self):
        """Test case where all caches are reused."""
        from src.data_utils import compute_cache_stats_per_split
        
        qa_df = pd.DataFrame([
            {"final_split": "train", "reused_cache": True},
            {"final_split": "train", "reused_cache": True},
            {"final_split": "validation", "reused_cache": True},
        ])
        
        stats = compute_cache_stats_per_split(qa_df)
        
        assert stats["train"]["fresh_count"] == 0
        assert stats["train"]["reuse_count"] == 2
        assert stats["validation"]["fresh_count"] == 0
        assert stats["validation"]["reuse_count"] == 1
    
    def test_different_ratios_per_split(self):
        """Splits can have different fresh/reuse ratios."""
        from src.data_utils import compute_cache_stats_per_split
        
        # Train: 4 fresh, 1 reused
        # Validation: 1 fresh, 4 reused
        qa_df = pd.DataFrame(
            [{"final_split": "train", "reused_cache": False}] * 4 +
            [{"final_split": "train", "reused_cache": True}] * 1 +
            [{"final_split": "validation", "reused_cache": False}] * 1 +
            [{"final_split": "validation", "reused_cache": True}] * 4
        )
        
        stats = compute_cache_stats_per_split(qa_df)
        
        assert stats["train"]["fresh_count"] == 4
        assert stats["train"]["reuse_count"] == 1
        assert stats["validation"]["fresh_count"] == 1
        assert stats["validation"]["reuse_count"] == 4
    
    def test_consistency_verification_passes(self):
        """Verify consistency check passes for valid stats."""
        from src.data_utils import compute_cache_stats_per_split, verify_cache_stats_consistency
        
        qa_df = pd.DataFrame([
            {"final_split": "train", "reused_cache": True},
            {"final_split": "train", "reused_cache": False},
            {"final_split": "validation", "reused_cache": True},
        ])
        
        stats = compute_cache_stats_per_split(qa_df)
        result = verify_cache_stats_consistency(stats, expected_totals={"train": 2, "validation": 1})
        
        assert result["passed"] is True
        assert result["global_fresh"] == 1
        assert result["global_reuse"] == 2
        assert result["global_total"] == 3
    
    def test_consistency_verification_fails_wrong_total(self):
        """Verify consistency check fails for wrong expected totals."""
        from src.data_utils import compute_cache_stats_per_split, verify_cache_stats_consistency
        
        qa_df = pd.DataFrame([
            {"final_split": "train", "reused_cache": True},
            {"final_split": "train", "reused_cache": False},
        ])
        
        stats = compute_cache_stats_per_split(qa_df)
        result = verify_cache_stats_consistency(stats, expected_totals={"train": 100})  # Wrong!
        
        assert result["passed"] is False
        assert len(result["errors"]) > 0
    
    def test_handles_string_boolean(self):
        """Test handling of string 'true'/'false' values."""
        from src.data_utils import compute_cache_stats_per_split
        
        qa_df = pd.DataFrame([
            {"final_split": "train", "reused_cache": "true"},
            {"final_split": "train", "reused_cache": "false"},
            {"final_split": "train", "reused_cache": "True"},
        ])
        
        stats = compute_cache_stats_per_split(qa_df)
        
        assert stats["train"]["reuse_count"] == 2
        assert stats["train"]["fresh_count"] == 1
    
    def test_handles_numpy_boolean(self):
        """Test handling of numpy boolean values."""
        from src.data_utils import compute_cache_stats_per_split
        
        qa_df = pd.DataFrame([
            {"final_split": "train", "reused_cache": np.bool_(True)},
            {"final_split": "train", "reused_cache": np.bool_(False)},
        ])
        
        stats = compute_cache_stats_per_split(qa_df)
        
        assert stats["train"]["reuse_count"] == 1
        assert stats["train"]["fresh_count"] == 1


# ============================================================================
# Validation contamination audit tests
# ============================================================================

class TestDataFrameContaminationAudit:
    """Tests for audit_dataframe_contamination (independent of vocabulary)."""
    
    def test_detects_khmer_in_text(self):
        """Should detect Khmer characters in normalized text."""
        from src.data_utils import audit_dataframe_contamination
        
        df = pd.DataFrame([
            {"record_uid": "u1", "text_bahnar": "hello ក world"},  # Khmer KA
            {"record_uid": "u2", "text_bahnar": "clean text"},
        ])
        
        char_df, row_df = audit_dataframe_contamination(df, split_name="validation")
        
        # Should find Khmer character
        assert len(char_df) >= 1
        assert "ក" in set(char_df["character"])
        
        # Should flag 1 row
        assert len(row_df) == 1
        assert row_df.iloc[0]["record_uid"] == "u1"
    
    def test_detects_thai_in_text(self):
        """Should detect Thai characters in normalized text."""
        from src.data_utils import audit_dataframe_contamination
        
        df = pd.DataFrame([
            {"record_uid": "u1", "text_bahnar": "test ก text"},  # Thai KO KAI
        ])
        
        char_df, row_df = audit_dataframe_contamination(df, split_name="validation")
        
        assert len(char_df) >= 1
        assert "ก" in set(char_df["character"])
        assert len(row_df) == 1
    
    def test_detects_cyrillic_in_text(self):
        """Should detect Cyrillic characters in normalized text."""
        from src.data_utils import audit_dataframe_contamination
        
        df = pd.DataFrame([
            {"record_uid": "u1", "text_bahnar": "test а text"},  # Cyrillic A (looks like Latin)
        ])
        
        char_df, row_df = audit_dataframe_contamination(df, split_name="validation")
        
        assert len(char_df) >= 1
        assert "а" in set(char_df["character"])
        assert len(row_df) == 1
    
    def test_does_not_flag_clean_text(self):
        """Clean Bahnar/Vietnamese text should not be flagged."""
        from src.data_utils import audit_dataframe_contamination
        
        df = pd.DataFrame([
            {"record_uid": "u1", "text_bahnar": "Pơlei Bahnar"},  # Clean Bahnar
            {"record_uid": "u2", "text_bahnar": "Xin chào"},  # Clean Vietnamese
            {"record_uid": "u3", "text_bahnar": "abc 123"},  # Basic Latin
        ])
        
        char_df, row_df = audit_dataframe_contamination(df, split_name="validation")
        
        # Should find no unexpected characters
        # char_df may be empty (no columns) if nothing flagged
        if not char_df.empty and "flags" in char_df.columns:
            unexpected = char_df.loc[char_df["flags"].str.contains("unexpected_script", na=False)]
            assert len(unexpected) == 0
        else:
            assert len(char_df) == 0  # No contaminated chars at all
        assert len(row_df) == 0
    
    def test_row_report_columns(self):
        """Row report should have all required columns."""
        from src.data_utils import audit_dataframe_contamination
        
        df = pd.DataFrame([
            {"record_uid": "u1", "record_id": "r1", "source_label": "s1",
             "group_id": "g1", "recording_group_id": "rg1",
             "text_bahnar": "test ក text"},
        ])
        
        char_df, row_df = audit_dataframe_contamination(df, split_name="validation")
        
        required = ["record_uid", "record_id", "source_label", "group_id",
                    "recording_group_id", "split", "text_bahnar", "text_bahnar_norm",
                    "flagged_characters", "flagged_codepoints", "flagged_unicode_names",
                    "flags", "contamination_ratio", "flag_reason"]
        for col in required:
            assert col in row_df.columns, f"Missing column: {col}"
    
    def test_char_report_has_occurrence_counts(self):
        """Character report should have occurrence and affected row counts."""
        from src.data_utils import audit_dataframe_contamination
        
        df = pd.DataFrame([
            {"record_uid": "u1", "text_bahnar": "ក ក ក"},  # 3 occurrences in 1 row
            {"record_uid": "u2", "text_bahnar": "ក"},  # 1 occurrence in another row
        ])
        
        char_df, row_df = audit_dataframe_contamination(df, split_name="validation")
        
        khmer_row = char_df.loc[char_df["character"] == "ក"]
        assert len(khmer_row) == 1
        assert khmer_row.iloc[0]["occurrence_count"] == 4  # 3 + 1
        assert khmer_row.iloc[0]["affected_row_count"] == 2


# ============================================================================
# Exclusion list and clean split tests
# ============================================================================

class TestExclusionListAndCleanSplit:
    """Tests for build_exclusion_list and build_clean_split."""
    
    def test_build_exclusion_list_from_contamination(self):
        """Should build exclusion list from contamination report."""
        from src.data_utils import audit_dataframe_contamination, build_exclusion_list
        
        df = pd.DataFrame([
            {"record_uid": "u1", "text_bahnar": "hello ក world"},  # Contaminated
            {"record_uid": "u2", "text_bahnar": "clean text"},  # Clean
            {"record_uid": "u3", "text_bahnar": "more ក text"},  # Contaminated
        ])
        
        _, row_df = audit_dataframe_contamination(df)
        exclusion_df = build_exclusion_list(row_df)
        
        assert len(exclusion_df) == 2
        assert set(exclusion_df["record_uid"]) == {"u1", "u3"}
    
    def test_build_clean_split_removes_contaminated(self):
        """Should remove contaminated rows from original DataFrame."""
        from src.data_utils import build_clean_split
        
        original_df = pd.DataFrame([
            {"record_uid": "u1", "text_bahnar": "hello"},
            {"record_uid": "u2", "text_bahnar": "world"},
            {"record_uid": "u3", "text_bahnar": "test"},
        ])
        
        exclusion_df = pd.DataFrame([
            {"record_uid": "u2", "flags": "unexpected_script", "exclusion_reason": "test"},
        ])
        
        clean_df = build_clean_split(original_df, exclusion_df)
        
        assert len(clean_df) == 2
        assert set(clean_df["record_uid"]) == {"u1", "u3"}
    
    def test_clean_split_preserves_all_if_no_exclusions(self):
        """Empty exclusion list should return original DataFrame."""
        from src.data_utils import build_clean_split
        
        original_df = pd.DataFrame([
            {"record_uid": "u1", "text_bahnar": "hello"},
            {"record_uid": "u2", "text_bahnar": "world"},
        ])
        
        exclusion_df = pd.DataFrame(columns=["record_uid", "flags", "exclusion_reason"])
        
        clean_df = build_clean_split(original_df, exclusion_df)
        
        assert len(clean_df) == 2
    
    def test_verify_clean_split_passes_for_clean_data(self):
        """Verification should pass for truly clean data."""
        from src.data_utils import verify_clean_split_no_contamination
        
        clean_df = pd.DataFrame([
            {"record_uid": "u1", "text_bahnar": "hello world"},
            {"record_uid": "u2", "text_bahnar": "Pơlei Bahnar"},
        ])
        
        result = verify_clean_split_no_contamination(clean_df, split_name="train_clean")
        
        assert result["passed"] is True
        assert result["unexpected_script_characters"] == 0
        assert result["unexpected_script_rows"] == 0
    
    def test_verify_clean_split_fails_if_contaminated(self):
        """Verification should fail if contamination remains."""
        from src.data_utils import verify_clean_split_no_contamination
        
        # Oops, this "clean" split still has Khmer
        clean_df = pd.DataFrame([
            {"record_uid": "u1", "text_bahnar": "hello ក world"},
        ])
        
        result = verify_clean_split_no_contamination(clean_df, split_name="train_clean")
        
        assert result["passed"] is False
        assert result["unexpected_script_characters"] > 0
        assert result["unexpected_script_rows"] > 0


# ============================================================================
# HTTP retry tests for download_audio_bytes_with_redirects
# ============================================================================

class TestDownloadWithRetry:
    """Tests for download_audio_bytes_with_redirects with retry logic."""
    
    def test_retry_on_429(self):
        """Should retry on HTTP 429 and succeed."""
        call_count = {"n": 0}
        sleeps = []
        
        class FakeResp:
            def __init__(self, status, content=b""):
                self.status_code = status
                self.headers = {}
                self._content = content
            
            def raise_for_status(self):
                if self.status_code >= 400:
                    import requests
                    err = requests.HTTPError(f"{self.status_code}")
                    err.response = self
                    raise err
            
            def iter_content(self, chunk_size=None):
                return iter([self._content])
            
            def close(self):
                pass
        
        class FakeSession:
            def request(self, *args, **kwargs):
                call_count["n"] += 1
                if call_count["n"] < 3:
                    return FakeResp(429)
                return FakeResp(200, b"AUDIODATA")
        
        def mock_resolver(host):
            return [ipaddress.ip_address("8.8.8.8")]
        
        data = download_audio_bytes_with_redirects(
            "https://cdn.example/audio.wav",
            session=FakeSession(),
            sleep_fn=lambda s: sleeps.append(s),
            dns_resolver=mock_resolver,
            max_attempts=6,
        )
        
        assert data == b"AUDIODATA"
        assert call_count["n"] == 3  # 2 failures + 1 success
        assert len(sleeps) == 2  # slept twice
    
    def test_retry_on_503(self):
        """Should retry on HTTP 503 and succeed."""
        call_count = {"n": 0}
        
        class FakeResp:
            def __init__(self, status, content=b""):
                self.status_code = status
                self.headers = {}
                self._content = content
            
            def raise_for_status(self):
                pass
            
            def iter_content(self, chunk_size=None):
                return iter([self._content])
            
            def close(self):
                pass
        
        class FakeSession:
            def request(self, *args, **kwargs):
                call_count["n"] += 1
                if call_count["n"] < 2:
                    return FakeResp(503)
                return FakeResp(200, b"OK")
        
        def mock_resolver(host):
            return [ipaddress.ip_address("8.8.8.8")]
        
        data = download_audio_bytes_with_redirects(
            "https://cdn.example/audio.wav",
            session=FakeSession(),
            sleep_fn=lambda s: None,
            dns_resolver=mock_resolver,
            max_attempts=6,
        )
        
        assert data == b"OK"
        assert call_count["n"] == 2
    
    def test_no_retry_on_404(self):
        """Should NOT retry on HTTP 404."""
        call_count = {"n": 0}
        
        class FakeResp:
            status_code = 404
            headers = {}
            _content = b""
            
            def raise_for_status(self):
                import requests
                err = requests.HTTPError("404")
                err.response = self
                raise err
            
            def iter_content(self, chunk_size=None):
                return iter([self._content])
            
            def close(self):
                pass
        
        class FakeSession:
            def request(self, *args, **kwargs):
                call_count["n"] += 1
                return FakeResp()
        
        def mock_resolver(host):
            return [ipaddress.ip_address("8.8.8.8")]
        
        with pytest.raises(RuntimeError, match="permanent|404"):
            download_audio_bytes_with_redirects(
                "https://cdn.example/notfound.wav",
                session=FakeSession(),
                sleep_fn=lambda s: None,
                dns_resolver=mock_resolver,
                max_attempts=6,
            )
        
        assert call_count["n"] == 1  # Only one attempt
    
    def test_redirect_with_retry_on_target(self):
        """Should follow redirect and then retry on target."""
        call_count = {"n": 0}
        
        class FakeResp:
            def __init__(self, status, location=None, content=b""):
                self.status_code = status
                self.headers = {"Location": location} if location else {}
                self._content = content
            
            def raise_for_status(self):
                pass
            
            def iter_content(self, chunk_size=None):
                return iter([self._content])
            
            def close(self):
                pass
        
        class FakeSession:
            def request(self, method, url, *args, **kwargs):
                call_count["n"] += 1
                if call_count["n"] == 1:
                    # First request: redirect
                    return FakeResp(302, "https://cdn2.example/actual.wav")
                if call_count["n"] == 2:
                    # Second request: 503 (retry)
                    return FakeResp(503)
                # Third request: success
                return FakeResp(200, content=b"AUDIO")
        
        def mock_resolver(host):
            return [ipaddress.ip_address("8.8.8.8")]
        
        data = download_audio_bytes_with_redirects(
            "https://cdn.example/audio.wav",
            session=FakeSession(),
            sleep_fn=lambda s: None,
            dns_resolver=mock_resolver,
            max_attempts=6,
        )
        
        assert data == b"AUDIO"
        assert call_count["n"] == 3  # redirect + retry + success
    
    def test_connection_error_retry(self):
        """Should retry on connection errors."""
        call_count = {"n": 0}
        
        class FakeResp:
            status_code = 200
            headers = {}
            _content = b"SUCCESS"
            
            def raise_for_status(self):
                pass
            
            def iter_content(self, chunk_size=None):
                return iter([self._content])
            
            def close(self):
                pass
        
        class FakeSession:
            def request(self, *args, **kwargs):
                call_count["n"] += 1
                if call_count["n"] < 3:
                    raise ConnectionError("Network unreachable")
                return FakeResp()
        
        def mock_resolver(host):
            return [ipaddress.ip_address("8.8.8.8")]
        
        data = download_audio_bytes_with_redirects(
            "https://cdn.example/audio.wav",
            session=FakeSession(),
            sleep_fn=lambda s: None,
            dns_resolver=mock_resolver,
            max_attempts=6,
        )
        
        assert data == b"SUCCESS"
        assert call_count["n"] == 3


# ============================================================================
# UID Hash tests (compute_ordered_uid_hash, compute_uid_set_hash)
# ============================================================================

class TestUidHashFunctions:
    """Tests for compute_ordered_uid_hash and compute_uid_set_hash."""
    
    def test_ordered_hash_preserves_order(self):
        """[a,b,c] and [c,b,a] must have different ordered hashes."""
        from src.data_utils import compute_ordered_uid_hash
        
        df1 = pd.DataFrame({"record_uid": ["a", "b", "c"]})
        df2 = pd.DataFrame({"record_uid": ["c", "b", "a"]})
        
        hash1 = compute_ordered_uid_hash(df1)
        hash2 = compute_ordered_uid_hash(df2)
        
        assert hash1 != hash2
        assert is_valid_sha256(hash1)
        assert is_valid_sha256(hash2)
    
    def test_set_hash_ignores_order(self):
        """[a,b,c] and [c,b,a] must have same set hash."""
        from src.data_utils import compute_uid_set_hash
        
        df1 = pd.DataFrame({"record_uid": ["a", "b", "c"]})
        df2 = pd.DataFrame({"record_uid": ["c", "b", "a"]})
        
        hash1 = compute_uid_set_hash(df1)
        hash2 = compute_uid_set_hash(df2)
        
        assert hash1 == hash2
        assert is_valid_sha256(hash1)
    
    def test_empty_df_returns_valid_sha256(self):
        """Empty DataFrame must return valid 64-char SHA-256, not 'empty'."""
        from src.data_utils import compute_ordered_uid_hash, compute_uid_set_hash
        
        df = pd.DataFrame({"record_uid": []})
        
        ordered = compute_ordered_uid_hash(df)
        set_hash = compute_uid_set_hash(df)
        
        assert is_valid_sha256(ordered)
        assert is_valid_sha256(set_hash)
        assert ordered == set_hash  # Both hash empty string
        assert ordered != "empty"
    
    def test_null_uid_raises_error(self):
        """Null/NaN in record_uid must raise ValueError."""
        from src.data_utils import compute_ordered_uid_hash, compute_uid_set_hash
        
        df = pd.DataFrame({"record_uid": ["a", None, "c"]})
        
        with pytest.raises(ValueError, match="null"):
            compute_ordered_uid_hash(df)
        
        with pytest.raises(ValueError, match="null"):
            compute_uid_set_hash(df)
    
    def test_duplicate_uid_raises_error(self):
        """Duplicate record_uid must raise ValueError."""
        from src.data_utils import compute_ordered_uid_hash, compute_uid_set_hash
        
        df = pd.DataFrame({"record_uid": ["a", "b", "a"]})
        
        with pytest.raises(ValueError, match="duplicate"):
            compute_ordered_uid_hash(df)
        
        with pytest.raises(ValueError, match="duplicate"):
            compute_uid_set_hash(df)
    
    def test_missing_column_raises_error(self):
        """Missing record_uid column must raise ValueError."""
        from src.data_utils import compute_ordered_uid_hash, compute_uid_set_hash
        
        df = pd.DataFrame({"other_col": ["a", "b", "c"]})
        
        with pytest.raises(ValueError, match="not found"):
            compute_ordered_uid_hash(df)
        
        with pytest.raises(ValueError, match="not found"):
            compute_uid_set_hash(df)


# ============================================================================
# CSV sidecar metadata tests
# ============================================================================

class TestWriteDataframeCsvMetadata:
    """Tests for enhanced write_dataframe_csv sidecar metadata."""
    
    def test_sidecar_has_row_count(self, tmp_path: Path):
        """Sidecar must have row_count matching saved DataFrame."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        path = tmp_path / "test.csv"
        
        write_dataframe_csv(df, path, metadata={"run_id": "test", "dataset_revision": "abc"})
        
        meta_path = path.with_suffix(".csv.meta.json")
        meta = json.loads(meta_path.read_text())
        
        assert meta["row_count"] == 3
    
    def test_sidecar_has_columns_in_order(self, tmp_path: Path):
        """Sidecar must have columns list preserving order."""
        df = pd.DataFrame({"z_col": [1], "a_col": [2], "m_col": [3]})
        path = tmp_path / "test.csv"
        
        write_dataframe_csv(df, path, metadata={"run_id": "test", "dataset_revision": "abc"})
        
        meta = json.loads(path.with_suffix(".csv.meta.json").read_text())
        
        assert meta["columns"] == ["z_col", "a_col", "m_col"]
    
    def test_sidecar_has_dtypes(self, tmp_path: Path):
        """Sidecar must have dtypes dict."""
        df = pd.DataFrame({"int_col": [1, 2], "str_col": ["a", "b"]})
        path = tmp_path / "test.csv"
        
        write_dataframe_csv(df, path, metadata={"run_id": "test", "dataset_revision": "abc"})
        
        meta = json.loads(path.with_suffix(".csv.meta.json").read_text())
        
        assert "dtypes" in meta
        assert "int_col" in meta["dtypes"]
        assert "str_col" in meta["dtypes"]
    
    def test_sidecar_has_file_sha256(self, tmp_path: Path):
        """Sidecar must have file_sha256 computed after write."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        path = tmp_path / "test.csv"
        
        write_dataframe_csv(df, path, metadata={"run_id": "test", "dataset_revision": "abc"})
        
        meta = json.loads(path.with_suffix(".csv.meta.json").read_text())
        
        assert is_valid_sha256(meta["file_sha256"])
        assert meta["file_sha256"] == sha256_file(path)
    
    def test_empty_csv_still_has_valid_schema(self, tmp_path: Path):
        """Empty CSV (0 rows) must still have valid header/schema."""
        df = pd.DataFrame(columns=["col1", "col2", "col3"])
        path = tmp_path / "empty.csv"
        
        write_dataframe_csv(df, path, metadata={"run_id": "test", "dataset_revision": "abc"})
        
        meta = json.loads(path.with_suffix(".csv.meta.json").read_text())
        
        assert meta["row_count"] == 0
        assert meta["columns"] == ["col1", "col2", "col3"]
        
        # CSV should be readable without EmptyDataError
        df_reloaded = pd.read_csv(path)
        assert list(df_reloaded.columns) == ["col1", "col2", "col3"]
        assert len(df_reloaded) == 0


class TestVerifyReportMetadataEnhanced:
    """Tests for enhanced verify_report_metadata."""
    
    def test_detects_tampered_csv(self, tmp_path: Path):
        """Modified CSV after sidecar creation must fail verification."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        path = tmp_path / "test.csv"
        
        write_dataframe_csv(df, path, metadata={
            "run_id": "test",
            "dataset_revision": "abc",
            "processing_version": "v1",
        })
        
        # Tamper with CSV
        path.write_text("col\n9\n9\n9\n")
        
        # Verification should fail
        assert verify_report_metadata(
            path,
            expected_run_id="test",
            expected_revision="abc",
            verify_sha=True,
        ) is False
    
    def test_detects_wrong_row_count(self, tmp_path: Path):
        """Wrong row count in sidecar must fail verification."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        path = tmp_path / "test.csv"
        
        write_dataframe_csv(df, path, metadata={
            "run_id": "test",
            "dataset_revision": "abc",
        })
        
        # Modify meta to have wrong row_count
        meta_path = path.with_suffix(".csv.meta.json")
        meta = json.loads(meta_path.read_text())
        meta["row_count"] = 999
        meta_path.write_text(json.dumps(meta))
        
        result = verify_report_metadata(
            path,
            expected_run_id="test",
            expected_revision="abc",
            verify_schema=True,
        )
        assert result is False
    
    def test_detects_wrong_run_id(self, tmp_path: Path):
        """Wrong run_id must fail verification."""
        df = pd.DataFrame({"col": [1]})
        path = tmp_path / "test.csv"
        
        write_dataframe_csv(df, path, metadata={"run_id": "old-run", "dataset_revision": "abc"})
        
        assert verify_report_metadata(
            path,
            expected_run_id="new-run",
            expected_revision="abc",
        ) is False


# ============================================================================
# Contamination audit fixed schema tests
# ============================================================================

class TestContaminationAuditFixedSchema:
    """Tests for audit_dataframe_contamination with fixed schema."""
    
    def test_clean_data_returns_empty_with_correct_schema(self):
        """Clean data returns empty DataFrames with correct column schemas."""
        from src.data_utils import audit_dataframe_contamination, CONTAMINATION_CHAR_SCHEMA, CONTAMINATION_ROW_SCHEMA
        
        clean_df = pd.DataFrame([
            {"record_uid": "u1", "text_bahnar": "hello world"},
            {"record_uid": "u2", "text_bahnar": "pơlei bahnar"},
        ])
        
        char_df, row_df = audit_dataframe_contamination(clean_df)
        
        assert len(char_df) == 0
        assert len(row_df) == 0
        assert list(char_df.columns) == CONTAMINATION_CHAR_SCHEMA
        assert list(row_df.columns) == CONTAMINATION_ROW_SCHEMA
    
    def test_empty_input_returns_correct_schema(self):
        """Empty input returns DataFrames with correct column schemas."""
        from src.data_utils import audit_dataframe_contamination, CONTAMINATION_CHAR_SCHEMA, CONTAMINATION_ROW_SCHEMA
        
        empty_df = pd.DataFrame(columns=["record_uid", "text_bahnar"])
        
        char_df, row_df = audit_dataframe_contamination(empty_df)
        
        assert len(char_df) == 0
        assert len(row_df) == 0
        assert list(char_df.columns) == CONTAMINATION_CHAR_SCHEMA
        assert list(row_df.columns) == CONTAMINATION_ROW_SCHEMA
    
    def test_csv_roundtrip_empty_report(self, tmp_path: Path):
        """Empty contamination report can be written and read without EmptyDataError."""
        from src.data_utils import audit_dataframe_contamination
        
        clean_df = pd.DataFrame([{"record_uid": "u1", "text_bahnar": "clean text"}])
        char_df, row_df = audit_dataframe_contamination(clean_df)
        
        char_path = tmp_path / "char_report.csv"
        row_path = tmp_path / "row_report.csv"
        
        char_df.to_csv(char_path, index=False)
        row_df.to_csv(row_path, index=False)
        
        # Should not raise EmptyDataError
        char_reloaded = pd.read_csv(char_path)
        row_reloaded = pd.read_csv(row_path)
        
        assert len(char_reloaded) == 0
        assert len(row_reloaded) == 0
    
    def test_hebrew_is_flagged_as_unexpected(self):
        """Hebrew characters should be flagged as unexpected_script."""
        from src.data_utils import classify_character, audit_dataframe_contamination
        
        # Hebrew letter ALEF
        assert classify_character("א") == "unexpected_script"
        
        df = pd.DataFrame([{"record_uid": "u1", "text_bahnar": "test א text"}])
        char_df, row_df = audit_dataframe_contamination(df)
        
        assert len(char_df) >= 1
        assert "א" in set(char_df["character"])
        assert len(row_df) == 1


# ============================================================================
# Clean split contract tests
# ============================================================================

class TestCleanSplitContract:
    """Tests for clean split contract export and verification."""
    
    def test_export_and_verify_roundtrip(self, tmp_path: Path):
        """Export contract and verify it matches."""
        from src.data_utils import (
            export_clean_split_contract, 
            verify_clean_split_contract,
            compute_ordered_uid_hash,
            compute_uid_set_hash,
        )
        
        # Create test data
        train_clean = pd.DataFrame({"record_uid": ["t1", "t2", "t3"]})
        val_clean = pd.DataFrame({"record_uid": ["v1", "v2"]})
        
        # Create exclusion files
        train_exc_path = tmp_path / "train_exc.csv"
        val_exc_path = tmp_path / "val_exc.csv"
        train_exc_path.write_text("record_uid\ne1\n")
        val_exc_path.write_text("record_uid\ne2\n")
        
        contract_path = tmp_path / "contract.json"
        
        contract = export_clean_split_contract(
            output_path=contract_path,
            run_id="test-run",
            dataset_id="test-dataset",
            dataset_revision="rev123",
            normalization_version="v1",
            base_manifest_sha256={"rq1_train.csv": "a"*64},
            train_original_count=4,
            train_exclusion_count=1,
            train_clean_count=3,
            train_ordered_uid_sha256=compute_ordered_uid_hash(train_clean),
            train_uid_set_sha256=compute_uid_set_hash(train_clean),
            train_exclusion_csv_sha256=sha256_file(train_exc_path),
            validation_original_count=3,
            validation_exclusion_count=1,
            validation_clean_count=2,
            validation_ordered_uid_sha256=compute_ordered_uid_hash(val_clean),
            validation_uid_set_sha256=compute_uid_set_hash(val_clean),
            validation_exclusion_csv_sha256=sha256_file(val_exc_path),
        )
        
        # Verify
        result = verify_clean_split_contract(
            contract_path,
            expected_run_id="test-run",
            expected_revision="rev123",
            train_clean_df=train_clean,
            validation_clean_df=val_clean,
            train_exclusion_path=train_exc_path,
            validation_exclusion_path=val_exc_path,
        )
        
        assert result["passed"] is True
    
    def test_wrong_run_id_fails_verification(self, tmp_path: Path):
        """Contract with wrong run_id fails verification."""
        contract_path = tmp_path / "contract.json"
        contract_path.write_text(json.dumps({
            "run_id": "old-run",
            "dataset_revision": "rev123",
            "train": {"clean_count": 0},
            "validation": {"clean_count": 0},
        }))
        
        result = verify_clean_split_contract(
            contract_path,
            expected_run_id="new-run",
            expected_revision="rev123",
            train_clean_df=pd.DataFrame({"record_uid": []}),
            validation_clean_df=pd.DataFrame({"record_uid": []}),
            train_exclusion_path=tmp_path / "t.csv",
            validation_exclusion_path=tmp_path / "v.csv",
        )
        
        assert result["passed"] is False
        assert result["run_id_match"] is False


# ============================================================================
# Tokenizer provenance tests
# ============================================================================

class TestTokenizerProvenance:
    """Tests for tokenizer provenance verification."""
    
    def test_exact_vocab_match_required(self, tmp_path: Path):
        """Tokenizer save must verify exact vocabulary mapping."""
        try:
            from transformers import Wav2Vec2CTCTokenizer
        except ImportError:
            pytest.skip("transformers not installed")
        
        from src.data_utils import save_tokenizer_clean
        
        vocab = {"[PAD]": 0, "[UNK]": 1, "|": 2, "a": 3, "b": 4, "ơ": 5}
        out_dir = tmp_path / "tokenizer"
        
        verification = save_tokenizer_clean(vocab, out_dir)
        
        assert verification["passed"] is True
        assert verification["vocab_exact_match"] is True
    
    def test_vocab_with_unexpected_script_fails(self, tmp_path: Path):
        """Vocabulary with unexpected scripts must fail verification."""
        try:
            from transformers import Wav2Vec2CTCTokenizer
        except ImportError:
            pytest.skip("transformers not installed")
        
        from src.data_utils import save_tokenizer_clean
        
        # Include Khmer character
        vocab = {"[PAD]": 0, "[UNK]": 1, "|": 2, "a": 3, "ក": 4}  # Khmer
        out_dir = tmp_path / "tokenizer"
        
        with pytest.raises(RuntimeError, match="unexpected"):
            save_tokenizer_clean(vocab, out_dir)
    
    def test_provenance_file_created(self, tmp_path: Path):
        """Provenance file is created when provenance dict provided."""
        try:
            from transformers import Wav2Vec2CTCTokenizer
        except ImportError:
            pytest.skip("transformers not installed")
        
        from src.data_utils import save_tokenizer_clean
        
        vocab = {"[PAD]": 0, "[UNK]": 1, "|": 2, "a": 3}
        out_dir = tmp_path / "tokenizer"
        
        provenance = {
            "run_id": "test-run",
            "dataset_id": "test",
            "dataset_revision": "rev123",
            "normalization_version": "v1",
            "train_clean_count": 100,
            "train_clean_ordered_uid_sha256": "a"*64,
            "train_clean_uid_set_sha256": "b"*64,
            "train_exclusion_csv_sha256": "c"*64,
        }
        
        save_tokenizer_clean(vocab, out_dir, provenance=provenance)
        
        prov_path = out_dir / "provenance.json"
        assert prov_path.is_file()
        
        prov_data = json.loads(prov_path.read_text())
        assert prov_data["run_id"] == "test-run"
        assert is_valid_sha256(prov_data["vocab_sha256"])


# ============================================================================
# HTTP downloader tests (streaming, size limits)
# ============================================================================

class TestHttpDownloaderStreaming:
    """Tests for HTTP downloader with streaming and size limits."""
    
    def test_content_length_exceeds_limit_blocked(self):
        """Content-Length exceeding limit must be blocked before reading."""
        from src.data_utils import ResponseTooLargeError
        
        class FakeResp:
            status_code = 200
            headers = {"Content-Length": "100000000"}  # 100 MB
            
            def iter_content(self, chunk_size=None):
                return iter([b"data"])
            
            def raise_for_status(self):
                pass
            
            def close(self):
                pass
        
        class FakeSession:
            def request(self, *args, **kwargs):
                return FakeResp()
        
        def mock_resolver(host):
            return [ipaddress.ip_address("8.8.8.8")]
        
        with pytest.raises(ResponseTooLargeError, match="Content-Length"):
            download_audio_bytes_with_redirects(
                "https://cdn.example/large.wav",
                session=FakeSession(),
                dns_resolver=mock_resolver,
                max_response_bytes=1024 * 1024,  # 1 MB
            )
    
    def test_streaming_content_exceeds_limit_blocked(self):
        """Streaming content exceeding limit must be blocked mid-stream."""
        from src.data_utils import ResponseTooLargeError
        
        class FakeResp:
            status_code = 200
            headers = {}  # No Content-Length
            
            def iter_content(self, chunk_size=None):
                # Return 2 MB in chunks
                for _ in range(20):
                    yield b"x" * (100 * 1024)  # 100 KB per chunk
            
            def raise_for_status(self):
                pass
            
            def close(self):
                pass
        
        class FakeSession:
            def request(self, *args, **kwargs):
                return FakeResp()
        
        def mock_resolver(host):
            return [ipaddress.ip_address("8.8.8.8")]
        
        with pytest.raises(ResponseTooLargeError, match="exceeds limit"):
            download_audio_bytes_with_redirects(
                "https://cdn.example/large.wav",
                session=FakeSession(),
                dns_resolver=mock_resolver,
                max_response_bytes=1024 * 1024,  # 1 MB limit
            )
    
    def test_response_closed_on_success(self):
        """Response must be closed on success."""
        closed = {"called": False}
        
        class FakeResp:
            status_code = 200
            headers = {}
            
            def iter_content(self, chunk_size=None):
                return iter([b"data"])
            
            def raise_for_status(self):
                pass
            
            def close(self):
                closed["called"] = True
        
        class FakeSession:
            def request(self, *args, **kwargs):
                return FakeResp()
        
        def mock_resolver(host):
            return [ipaddress.ip_address("8.8.8.8")]
        
        download_audio_bytes_with_redirects(
            "https://cdn.example/file.wav",
            session=FakeSession(),
            dns_resolver=mock_resolver,
        )
        
        assert closed["called"] is True
    
    def test_response_closed_on_redirect(self):
        """Response must be closed on redirect."""
        closed_count = {"n": 0}
        call_count = {"n": 0}
        
        class FakeResp:
            def __init__(self, status, location=None, content=b""):
                self.status_code = status
                self.headers = {"Location": location} if location else {}
                self._content = content
            
            def iter_content(self, chunk_size=None):
                return iter([self._content])
            
            def raise_for_status(self):
                pass
            
            def close(self):
                closed_count["n"] += 1
        
        class FakeSession:
            def request(self, *args, **kwargs):
                call_count["n"] += 1
                if call_count["n"] == 1:
                    return FakeResp(302, "https://cdn2.example/file.wav")
                return FakeResp(200, content=b"data")
        
        def mock_resolver(host):
            return [ipaddress.ip_address("8.8.8.8")]
        
        download_audio_bytes_with_redirects(
            "https://cdn.example/file.wav",
            session=FakeSession(),
            dns_resolver=mock_resolver,
        )
        
        assert closed_count["n"] == 2  # Redirect closed + success closed
    
    def test_no_retry_on_418(self):
        """HTTP 418 (I'm a teapot) should not be retried."""
        call_count = {"n": 0}
        
        class FakeResp:
            status_code = 418
            headers = {}
            
            def iter_content(self, chunk_size=None):
                return iter([])
            
            def raise_for_status(self):
                pass
            
            def close(self):
                pass
        
        class FakeSession:
            def request(self, *args, **kwargs):
                call_count["n"] += 1
                return FakeResp()
        
        def mock_resolver(host):
            return [ipaddress.ip_address("8.8.8.8")]
        
        with pytest.raises(RuntimeError, match="permanent"):
            download_audio_bytes_with_redirects(
                "https://cdn.example/teapot.wav",
                session=FakeSession(),
                dns_resolver=mock_resolver,
                max_attempts=6,
            )
        
        assert call_count["n"] == 1  # Only one attempt
    
    def test_cache_hit_no_http_call(self, tmp_path: Path):
        """Valid cache hit should not make any HTTP calls."""
        import soundfile as sf
        from src.data_utils import set_dns_resolver_for_testing
        
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        
        # Create valid cache
        sr = 16000
        audio = np.sin(np.linspace(0, 1, int(sr * 0.5))).astype(np.float32) * 0.5
        
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV")
        audio_bytes = buf.getvalue()
        
        http_called = {"n": 0}
        
        def fake_download(url):
            http_called["n"] += 1
            return audio_bytes
        
        def fake_decode(data):
            return sf.read(io.BytesIO(data), always_2d=False)
        
        set_dns_resolver_for_testing(lambda h: [ipaddress.ip_address("8.8.8.8")])
        try:
            # First call creates cache
            process_audio_candidate(
                record_uid="test-uid",
                record_id="r1",
                final_split="train",
                source_label="s1",
                group_id="g1",
                duration_seconds_meta=0.5,
                text_identity_ok=True,
                audio_src="https://example.com/audio.wav",
                row_idx=0,
                cache_dir=cache_dir,
                project_root=tmp_path,
                dataset_revision="rev123",
                target_sr=16000,
                min_duration=0.05,
                max_duration=120.0,
                download_fn=fake_download,
                decode_fn=fake_decode,
            )
            
            first_call_count = http_called["n"]
            
            # Second call should use cache
            result = process_audio_candidate(
                record_uid="test-uid",
                record_id="r1",
                final_split="train",
                source_label="s1",
                group_id="g1",
                duration_seconds_meta=0.5,
                text_identity_ok=True,
                audio_src="https://example.com/audio.wav",
                row_idx=0,
                cache_dir=cache_dir,
                project_root=tmp_path,
                dataset_revision="rev123",
                target_sr=16000,
                min_duration=0.05,
                max_duration=120.0,
                download_fn=fake_download,
                decode_fn=fake_decode,
            )
        finally:
            set_dns_resolver_for_testing(None)
        
        assert result["reused_cache"] is True
        assert http_called["n"] == first_call_count  # No additional HTTP calls
