"""
Read-only preflight for the pinned parquet snapshot, URL parsing and disk budgets.

No bulk parquet is downloaded here: the Hugging Face API is injected as a stub so
every branch (right SHA, wrong SHA, missing ref, missing shard) is exercised
offline.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.asr_full_data import (
    assert_local_disk_budget,
    assert_parquet_files_are_train_only,
    expected_wav_bytes,
    fetch_parquet_shard_sizes,
    normalize_parquet_ref,
    parse_hf_parquet_url,
    verify_pinned_parquet_snapshot,
    wav_bytes_for_samples,
)

REPO = "hoangbinhmta99/bahnar-speech"
PINNED = "ad0a84362053dad098b86d0df215eb081f891dd8"
OTHER = "b" * 40
SHARDS = [f"default/train/{i:04d}.parquet" for i in range(50)]


class _StubApi:
    """Minimal stand-in for HfApi covering only what the preflight calls."""

    def __init__(
        self, *, files=None, dynamic_sha=PINNED, revisions=(PINNED,), ref_error=None,
        sizes=None, info_error=None,
    ):
        self.files = list(files if files is not None else SHARDS)
        self.dynamic_sha = dynamic_sha
        self.revisions = set(revisions)
        self.ref_error = ref_error
        self.sizes = dict(sizes) if sizes is not None else {n: 1_400_000_000 for n in self.files}
        self.info_error = info_error
        self.calls = []

    def list_repo_files(self, repo_id, revision=None, repo_type=None):
        self.calls.append(("list", revision))
        if revision not in self.revisions:
            raise RuntimeError(f"404 revision {revision} not found")
        return self.files

    def repo_info(self, repo_id, revision=None, repo_type=None, files_metadata=False):
        self.calls.append(("info", revision, files_metadata))
        if files_metadata:
            if self.info_error:
                raise RuntimeError(self.info_error)
            siblings = [
                type("Sibling", (), {"rfilename": name, "size": self.sizes.get(name)})()
                for name in self.files
            ]
            return type("Info", (), {"sha": self.dynamic_sha, "siblings": siblings})()
        if self.ref_error:
            raise RuntimeError(self.ref_error)
        return type("Info", (), {"sha": self.dynamic_sha})()


class TestPinnedSnapshotVerification:
    def test_matching_pin_and_complete_shards_pass(self):
        api = _StubApi()
        out = verify_pinned_parquet_snapshot(
            repo_id=REPO, pinned_revision=PINNED, required_filenames=SHARDS, api=api,
        )
        assert out["pinned_revision"] == PINNED
        assert out["required_count"] == 50
        assert out["warnings"] == []
        assert out["dynamic_matches_pinned"] is True

    def test_moved_conversion_ref_is_only_a_warning(self):
        """Pinning a commit is precisely so a moving ref stops mattering."""
        api = _StubApi(dynamic_sha=OTHER)
        out = verify_pinned_parquet_snapshot(
            repo_id=REPO, pinned_revision=PINNED, required_filenames=SHARDS, api=api,
        )
        assert out["dynamic_matches_pinned"] is False
        assert len(out["warnings"]) == 1 and OTHER in out["warnings"][0]

    def test_unresolvable_conversion_ref_is_only_a_warning(self):
        api = _StubApi(ref_error="ref refs/convert/parquet does not exist")
        out = verify_pinned_parquet_snapshot(
            repo_id=REPO, pinned_revision=PINNED, required_filenames=SHARDS, api=api,
        )
        assert out["dynamic_revision"] is None
        assert "could not resolve" in out["warnings"][0]

    def test_unresolvable_pinned_sha_fails(self):
        api = _StubApi(revisions=(OTHER,))
        with pytest.raises(RuntimeError, match="does not resolve"):
            verify_pinned_parquet_snapshot(
                repo_id=REPO, pinned_revision=PINNED, required_filenames=SHARDS, api=api,
            )

    def test_missing_required_shard_fails(self):
        api = _StubApi(files=SHARDS[:-1])
        with pytest.raises(RuntimeError, match="missing 1 required shard"):
            verify_pinned_parquet_snapshot(
                repo_id=REPO, pinned_revision=PINNED, required_filenames=SHARDS, api=api,
            )

    def test_non_immutable_revision_label_is_rejected(self):
        """A moving label would let the audio bytes change under a valid contract."""
        for label in ("refs/convert/parquet", "main", "ad0a843"):
            with pytest.raises(RuntimeError, match="40-hex immutable commit SHA"):
                verify_pinned_parquet_snapshot(
                    repo_id=REPO, pinned_revision=label,
                    required_filenames=SHARDS, api=_StubApi(),
                )

    def test_no_bulk_download_happens(self):
        api = _StubApi()
        verify_pinned_parquet_snapshot(
            repo_id=REPO, pinned_revision=PINNED, required_filenames=SHARDS, api=api,
        )
        assert [c[0] for c in api.calls] == ["list", "info"]


class TestTrainOnlyManifests:
    def test_train_shards_are_accepted(self):
        assert assert_parquet_files_are_train_only(SHARDS[:3]) == SHARDS[:3]

    def test_frozen_split_parquet_is_refused(self):
        for leaked in ("default/test/0000.parquet", "default/validation/0000.parquet"):
            with pytest.raises(RuntimeError, match="outside 'default/train/'"):
                assert_parquet_files_are_train_only([SHARDS[0], leaked])

    def test_duplicates_collapse_without_reordering(self):
        assert assert_parquet_files_are_train_only(
            [SHARDS[1], SHARDS[0], SHARDS[1]]
        ) == [SHARDS[1], SHARDS[0]]


class TestHfUrlParsing:
    @pytest.mark.parametrize("url", [
        f"https://huggingface.co/datasets/{REPO}/resolve/{PINNED}/default/train/0007.parquet",
        f"https://huggingface.co/datasets/{REPO}/blob/{PINNED}/default/train/0007.parquet",
        f"https://huggingface.co/datasets/{REPO}/resolve/{PINNED}/default/train/0007.parquet?download=true",
        f"https://huggingface.co/datasets/{REPO}/resolve/{PINNED}/default%2Ftrain%2F0007.parquet",
    ])
    def test_every_url_form_yields_the_same_file_and_revision(self, url):
        ref = parse_hf_parquet_url(url)
        assert ref.filename == "default/train/0007.parquet"
        assert ref.revision == PINNED
        assert ref.repo_id == REPO and ref.repo_type == "dataset"

    def test_a_bare_filename_is_pinned_to_the_configured_revision(self):
        """Manifests store plain filenames; the pin comes from the contract."""
        ref = normalize_parquet_ref(
            "default/train/0007.parquet", expected_repo_id=REPO, parquet_revision=PINNED,
        )
        assert ref.filename == "default/train/0007.parquet"
        assert ref.revision == PINNED

    def test_a_url_never_overrides_the_pinned_revision(self):
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{OTHER}/default/train/0007.parquet"
        ref = normalize_parquet_ref(url, expected_repo_id=REPO, parquet_revision=PINNED)
        assert ref.revision == PINNED


class TestLocalDiskBudget:
    def test_wav_size_is_exact_for_pcm16_mono(self):
        assert wav_bytes_for_samples(16000) == 16000 * 2 + 44

    def test_budget_is_summed_from_eligible_records_not_a_guess(self):
        train = pd.DataFrame({"record_uid": ["a", "b"], "n_samples": [16000, 32000]})
        val = pd.DataFrame({"record_uid": ["b", "c"], "n_samples": [32000, 8000]})
        out = expected_wav_bytes([train, val])
        # "b" is shared by both splits and must be counted once.
        assert out["unique_records"] == 3
        assert out["n_samples"] == 56000
        assert out["wav_bytes"] == sum(wav_bytes_for_samples(n) for n in (16000, 32000, 8000))
        assert out["audio_hours"] == pytest.approx(56000 / 16000 / 3600)

    def test_frame_without_sample_counts_is_refused(self):
        with pytest.raises(RuntimeError, match="lacks n_samples"):
            expected_wav_bytes([pd.DataFrame({"record_uid": ["a"]})])

    def test_sufficient_disk_reports_the_itemised_budget(self, tmp_path: Path):
        out = assert_local_disk_budget(
            tmp_path, needs={"wav": 1024, "parquet": 2048},
            reserve_bytes=0, label="unit",
        )
        assert out["required_bytes"] == 3072
        assert out["needs"] == {"wav": 1024, "parquet": 2048}
        assert out["ok"] is True

    def test_insufficient_disk_fails_and_names_the_components(self, tmp_path: Path):
        with pytest.raises(RuntimeError) as exc:
            assert_local_disk_budget(
                tmp_path, needs={"hydrated_wav": 10 ** 15, "local_checkpoints": 10 ** 14},
                reserve_bytes=5 * 1024 ** 3, label="full_train",
            )
        message = str(exc.value)
        assert "full_train" in message
        assert "hydrated_wav" in message and "local_checkpoints" in message
        assert "reserve=" in message

    def test_reserve_alone_can_fail_the_budget(self, tmp_path: Path):
        with pytest.raises(RuntimeError, match="Insufficient local disk"):
            assert_local_disk_budget(
                tmp_path, needs={}, reserve_bytes=10 ** 18, label="reserve-only",
            )


class TestMeasuredShardSizes:
    """The disk preflight must budget from real shard sizes, never a constant."""

    def test_sizes_come_from_the_pinned_snapshot(self):
        api = _StubApi(sizes={n: 1_000 + i for i, n in enumerate(SHARDS)})
        sizes = fetch_parquet_shard_sizes(
            repo_id=REPO, revision=PINNED, filenames=SHARDS[:3], api=api,
        )
        assert sizes == {SHARDS[0]: 1_000, SHARDS[1]: 1_001, SHARDS[2]: 1_002}
        assert api.calls == [("info", PINNED, True)]

    def test_missing_size_is_fail_closed(self):
        api = _StubApi(sizes={SHARDS[0]: 5_000})  # the rest report no size
        with pytest.raises(RuntimeError, match="reported no size"):
            fetch_parquet_shard_sizes(
                repo_id=REPO, revision=PINNED, filenames=SHARDS[:2], api=api,
            )

    def test_zero_size_counts_as_missing(self):
        api = _StubApi(sizes={n: 0 for n in SHARDS})
        with pytest.raises(RuntimeError, match="reported no size"):
            fetch_parquet_shard_sizes(
                repo_id=REPO, revision=PINNED, filenames=SHARDS[:1], api=api,
            )

    def test_api_failure_blocks_prepare_instead_of_guessing(self):
        api = _StubApi(info_error="503 service unavailable")
        with pytest.raises(RuntimeError, match="refusing to start prepare"):
            fetch_parquet_shard_sizes(
                repo_id=REPO, revision=PINNED, filenames=SHARDS[:1], api=api,
            )
