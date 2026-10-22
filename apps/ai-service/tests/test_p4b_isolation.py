"""The integration guard refuses the development stack before any write."""

from __future__ import annotations

import uuid

import pytest

from app.indexing.isolation import (
    IsolationError,
    isolated_p4b_targets,
    prepare_isolated_target,
    require_compose_ports,
    require_test_collection,
)
from p4b_cleanup import OwnedResources, generation_delete_body, run_cleanups
from ocr_scan_fixture import LINES, build_scan_pdf
from app.indexing.ocr_smoke import missing_smoke_tokens
from pypdf import PdfReader
import io


COMPLETE = {
    "CAS_P4B_TEST_COMPOSE_PROJECT": "citizen-assistance-p4b-test",
    "CAS_P4B_TEST_DATABASE_URL": "postgres://cas:cas@127.0.0.1:5434/citizen_assistance_p4b_test?sslmode=disable",
    "CAS_P4B_TEST_BUCKET": "cas-p4b-integration-test",
    "CAS_P4B_TEST_MINIO_ENDPOINT": "127.0.0.1:9010",
    "CAS_P4B_TEST_MINIO_ACCESS_KEY": "test-key",
    "CAS_P4B_TEST_MINIO_SECRET_KEY": "test-secret",
    "CAS_P4B_TEST_QDRANT_URL": "http://127.0.0.1:6335",
}
PORTS = """
citizen-assistance-p4b-test-postgres-1 127.0.0.1:5434->5432/tcp
citizen-assistance-p4b-test-minio-1 127.0.0.1:9010->9000/tcp, 127.0.0.1:9011->9001/tcp
citizen-assistance-p4b-test-qdrant-1 127.0.0.1:6335->6333/tcp
"""
BASE = "postgres://cas:cas@127.0.0.1:5434/citizen_assistance_p4b_test?sslmode=disable"


def _rejected(env: dict[str, str]) -> None:
    calls: list[str] = []

    def inspect(_target: dict[str, str]) -> None:
        calls.append("inspect")

    with pytest.raises(IsolationError):
        prepare_isolated_target(env, inspect=inspect)
    assert calls == []


def test_guard_rejects_missing_targets():
    _rejected({})
    _rejected({"CAS_P4B_INTEGRATION": "1", "TEST_MODE": "true"})


def test_guard_rejects_development_and_unverified_targets():
    cases = []
    dev_db = dict(COMPLETE)
    dev_db["CAS_P4B_TEST_DATABASE_URL"] = "postgres://cas:cas@127.0.0.1:5432/citizen_assistance?sslmode=disable"
    cases.append(dev_db)
    dev_bucket = dict(COMPLETE)
    dev_bucket["CAS_P4B_TEST_BUCKET"] = "cas-documents"
    cases.append(dev_bucket)
    dev_minio = dict(COMPLETE)
    dev_minio["CAS_P4B_TEST_MINIO_ENDPOINT"] = "127.0.0.1:9000"
    cases.append(dev_minio)
    dev_qdrant = dict(COMPLETE)
    dev_qdrant["CAS_P4B_TEST_QDRANT_URL"] = "http://127.0.0.1:6333"
    cases.append(dev_qdrant)
    prod = dict(COMPLETE)
    prod["CAS_P4B_TEST_DATABASE_URL"] = "postgres://cas:cas@prod-db.internal:5432/citizen_assistance_prod?sslmode=disable"
    prod["CAS_P4B_TEST_MINIO_ENDPOINT"] = "storage.prod.internal:9000"
    prod["CAS_P4B_TEST_BUCKET"] = "citizen-prod-documents"
    prod["CAS_P4B_TEST_QDRANT_URL"] = "https://qdrant.prod.internal"
    cases.append(prod)
    name_only = dict(COMPLETE)
    name_only["CAS_P4B_TEST_DATABASE_URL"] = "postgres://cas:cas@prod-db.internal:5434/citizen_assistance_test?sslmode=disable"
    cases.append(name_only)
    wrong_port = dict(COMPLETE)
    wrong_port["CAS_P4B_TEST_DATABASE_URL"] = "postgres://cas:cas@127.0.0.1:5432/citizen_assistance_p4b_test?sslmode=disable"
    cases.append(wrong_port)
    marker = dict(COMPLETE)
    marker["CAS_P4B_TEST_COMPOSE_PROJECT"] = "citizen-assistance"
    cases.append(marker)
    for suffix in (
        "&host=prod-db.internal",
        "&hostaddr=10.0.0.8",
        "&port=5432",
        "&dbname=citizen_assistance",
        "&service=prod",
        "&servicefile=/tmp/pg_service.conf",
        "&sslmode=disable",
        "&host=/var/run/postgresql",
    ):
        overridden = dict(COMPLETE)
        overridden["CAS_P4B_TEST_DATABASE_URL"] = BASE + suffix
        cases.append(overridden)
    multi = dict(COMPLETE)
    multi["CAS_P4B_TEST_DATABASE_URL"] = "postgres://cas:cas@127.0.0.1:5434,prod-db.internal:5432/citizen_assistance_p4b_test?sslmode=disable"
    cases.append(multi)
    socket_url = dict(COMPLETE)
    socket_url["CAS_P4B_TEST_DATABASE_URL"] = "postgres://cas:cas@/citizen_assistance_p4b_test?sslmode=disable"
    cases.append(socket_url)
    for env in cases:
        _rejected(env)
    with pytest.raises(IsolationError):
        require_test_collection("knowledge_chunks")
    with pytest.raises(IsolationError):
        require_compose_ports("citizen-assistance-postgres-1 0.0.0.0:5432->5432/tcp\n")
    with pytest.raises(IsolationError):
        require_compose_ports(PORTS.replace("127.0.0.1:", "0.0.0.0:"))


def test_guard_accepts_only_the_pinned_test_stack():
    seen: list[str] = []

    def inspect(target: dict[str, str]) -> None:
        seen.append(target["database_url"])
        require_compose_ports(PORTS)

    target = prepare_isolated_target(COMPLETE, inspect=inspect)
    locked = COMPLETE["CAS_P4B_TEST_DATABASE_URL"] + "&hostaddr=127.0.0.1"
    assert seen == [locked]
    assert target["database_url"] == locked
    assert target["bucket"] == "cas-p4b-integration-test"
    assert target["qdrant_url"] == "http://127.0.0.1:6335"
    assert require_test_collection("p4b_test") == "p4b_test"
    assert require_test_collection("p4b_e5") == "p4b_e5"
    assert isolated_p4b_targets(COMPLETE)["compose_project"] == "citizen-assistance-p4b-test"


def test_scan_fixture_has_no_text_layer_and_keywords_match_its_lines():
    data = build_scan_pdf()
    reader = PdfReader(io.BytesIO(data))
    assert (reader.pages[0].extract_text() or "").strip() == ""
    assert missing_smoke_tokens("\n".join(LINES)) == []


def test_scan_fixture_builds_without_system_font(monkeypatch):
    import ocr_scan_fixture as fixture

    monkeypatch.setattr(fixture, "FONT_CANDIDATES", ())
    data = fixture.build_scan_pdf()
    reader = PdfReader(io.BytesIO(data))
    assert data.startswith(b"%PDF")
    assert (reader.pages[0].extract_text() or "").strip() == ""


def test_cleanup_keeps_going_and_only_names_owned_generations():
    owned = uuid.uuid4()
    existing = uuid.uuid4()
    body = generation_delete_body(owned)
    assert str(existing) not in str(body)
    assert body["filter"]["must"] == [{"key": "generation_id", "match": {"value": str(owned)}}]
    calls: list[str] = []

    def fail_once() -> None:
        calls.append("fail")
        raise RuntimeError("qdrant down")

    def succeed() -> None:
        calls.append("succeed")

    errors = run_cleanups([fail_once, succeed, succeed])
    assert [type(item).__name__ for item in errors] == ["RuntimeError"]
    assert calls == ["fail", "succeed", "succeed"]


def test_lock_database_url_uses_the_right_query_separator(monkeypatch):
    from psycopg.conninfo import conninfo_to_dict
    from psycopg.pq import PGconn

    from app.indexing.isolation import lock_database_url

    bare = "postgres://cas:cas@127.0.0.1:5434/citizen_assistance_p4b_test"
    locked_bare = lock_database_url(bare)
    locked_query = lock_database_url(BASE)
    assert locked_bare == bare + "?hostaddr=127.0.0.1"
    assert locked_query == BASE + "&hostaddr=127.0.0.1"
    monkeypatch.setenv("PGHOSTADDR", "127.0.0.2")
    for url in (locked_bare, locked_query):
        info = conninfo_to_dict(url)
        assert info["dbname"] == "citizen_assistance_p4b_test"
        assert info["hostaddr"] == "127.0.0.1"
        assert str(info["port"]) == "5434"
        conn = PGconn.connect_start(url.encode())
        try:
            assert conn.db == b"citizen_assistance_p4b_test"
            assert conn.hostaddr == b"127.0.0.1"
            assert conn.port == b"5434"
        finally:
            conn.finish()


def test_libpq_environment_overrides_are_rejected(monkeypatch):
    from psycopg.pq import PGconn

    from app.indexing.isolation import lock_database_url

    monkeypatch.setenv("PGHOSTADDR", "127.0.0.2")
    conn = PGconn.connect_start(BASE.encode())
    try:
        assert conn.host == b"127.0.0.1"
        assert conn.hostaddr == b"127.0.0.2"
    finally:
        conn.finish()
    _rejected(COMPLETE)
    monkeypatch.delenv("PGHOSTADDR", raising=False)
    locked = lock_database_url(BASE)
    monkeypatch.setenv("PGHOSTADDR", "127.0.0.2")
    conn = PGconn.connect_start(locked.encode())
    try:
        assert conn.hostaddr == b"127.0.0.1"
        assert conn.port == b"5434"
        assert conn.db == b"citizen_assistance_p4b_test"
    finally:
        conn.finish()


def test_service_file_and_other_postgres_env_are_rejected(monkeypatch, tmp_path):
    from psycopg.pq import PGconn

    service = tmp_path / "pg_service.conf"
    service.write_text("[probe]\nhostaddr=127.0.0.3\n", encoding="utf-8")
    monkeypatch.setenv("PGSERVICE", "probe")
    monkeypatch.setenv("PGSERVICEFILE", str(service))
    conn = PGconn.connect_start(BASE.encode())
    try:
        assert conn.hostaddr == b"127.0.0.3"
    finally:
        conn.finish()
    _rejected(COMPLETE)
    monkeypatch.delenv("PGSERVICE", raising=False)
    monkeypatch.delenv("PGSERVICEFILE", raising=False)
    for name, value in (
        ("PGHOST", "10.0.0.8"),
        ("PGPORT", "5432"),
        ("PGDATABASE", "citizen_assistance"),
        ("PGSERVICEFILE", str(service)),
        ("PGSYSCONFDIR", str(tmp_path)),
    ):
        monkeypatch.setenv(name, value)
        _rejected(COMPLETE)
        monkeypatch.delenv(name, raising=False)


def test_early_failure_cleans_only_resources_that_were_created():
    removed: list[str] = []
    existing = {"object-before", "row-before", "point-before"}

    def remember(name: str):
        def action() -> None:
            if name in existing:
                raise AssertionError(name)
            removed.append(name)
        return action

    upload_failed = OwnedResources()
    try:
        raise RuntimeError("upload failed")
    except RuntimeError as exc:
        assert upload_failed.finish(exc) == []
    assert removed == []

    insert_failed = OwnedResources()
    try:
        insert_failed.own(remember("object"))
        raise RuntimeError("insert failed")
    except RuntimeError as exc:
        assert insert_failed.finish(exc) == []
    assert removed == ["object"]

    qdrant_failed = OwnedResources()
    try:
        qdrant_failed.own(remember("object"))
        qdrant_failed.own(remember("rows"))
        qdrant_failed.own(remember("points"))
        raise RuntimeError("qdrant upsert failed")
    except RuntimeError as exc:
        assert qdrant_failed.finish(exc) == []
    assert removed == ["object", "points", "rows", "object"]

    assertion_failed = OwnedResources()
    try:
        assertion_failed.own(remember("object"))
        assertion_failed.own(remember("rows"))
        assertion_failed.own(remember("points"))
        raise AssertionError("count mismatch")
    except AssertionError as exc:
        assert assertion_failed.finish(exc) == []
    assert "object-before" not in removed
    assert removed[-3:] == ["points", "rows", "object"]

    repeated = OwnedResources()
    repeated.own(remember("once"))
    assert repeated.finish(None) == []
    assert repeated.finish(None) == []
    assert removed.count("once") == 1


def test_ambiguous_success_is_cleaned_because_the_id_was_armed_first():
    removed: list[str] = []
    persisted: list[str] = []
    preexisting = "object-before"

    def cleanup(name: str):
        def action() -> None:
            if name == preexisting:
                raise AssertionError(name)
            removed.append(name)
        return action

    owned = OwnedResources()
    owned.own(cleanup("object"))
    owned.own(cleanup("rows"))
    owned.own(cleanup("points"))
    try:
        persisted.append("object")
        raise TimeoutError("minio stored the object before the client returned")
    except TimeoutError as exc:
        assert owned.finish(exc) == []
    assert persisted == ["object"]
    assert removed == ["points", "rows", "object"]
    assert preexisting not in removed
    assert owned.finish(None) == []


def test_cleanup_failure_stays_visible_beside_the_test_error():
    warned = OwnedResources()

    def broken() -> None:
        raise RuntimeError("cleanup failed")

    warned.own(broken)
    try:
        raise AssertionError("assertion failed")
    except AssertionError as exc:
        with pytest.warns(RuntimeWarning, match="CLEANUP_FAILED"):
            errors = warned.finish(exc)
        assert len(errors) == 1
        assert isinstance(exc, AssertionError)
    assert warned.finish(None) == []

    alone = OwnedResources()
    alone.own(broken)
    with pytest.raises(RuntimeError, match="cleanup failed"):
        alone.finish(None)


def test_keyword_check_does_not_hide_a_wrong_reading():
    assert "giay" in missing_smoke_tokens("khai sinh gia lai so 123")
    assert missing_smoke_tokens("Giay   khai sinh, Gia Lai. So: 123") == []
