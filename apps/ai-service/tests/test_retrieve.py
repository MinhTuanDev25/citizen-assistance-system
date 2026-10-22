"""Phase 5 retrieve: embedding-compatible search, scoped filters, fail-closed support."""

from __future__ import annotations

import hashlib
import uuid

import pytest
from fastapi.testclient import TestClient

from app.indexing.embed import E5_DIMENSION, QUERY_PREFIX
from app.indexing.pipeline import FakeEmbedder
from app.indexing.retrieve import (
    STATUS_INCOMPATIBLE,
    STATUS_NONE,
    STATUS_OK,
    STATUS_REJECTED,
    RetrieveError,
    classify_support,
    fake_embedder_identity,
    has_query_overlap,
    has_related_overlap,
    passes_related_floor,
    qdrant_filter,
    query_embedder,
    reset_query_embedder_cache,
    run_retrieve,
    valid_page_range,
)
from app.indexing.support import classify_support as map_classify_support
from app.main import app
from app.models.retrieve import MAX_ALLOWED_SOURCES, RetrieveRequest, RetrieveSource
from tests.conftest import AUTH_HEADER

client = TestClient(app)

DOC = "11111111-1111-1111-1111-111111111111"
GEN = "22222222-2222-2222-2222-222222222222"
VER = "33333333-3333-3333-3333-333333333333"
CHUNK = "44444444-4444-4444-4444-444444444444"
OTHER = "55555555-5555-5555-5555-555555555555"


def _req(**kwargs) -> RetrieveRequest:
    body = {
        "schema_version": "retrieve.v1",
        "query": "Tôi muốn đăng ký khai sinh",
        "xa_id": "xa_chu_se",
        "procedure_id": DOC,
        "procedure_version_id": VER,
        "allowed_sources": [RetrieveSource(document_id=DOC, generation_id=GEN)],
        "top_k": 3,
        "statements": ["Bản chính giấy chứng sinh do cơ sở y tế cấp"],
    }
    body.update(kwargs)
    if "allowed_sources" in kwargs and isinstance(kwargs["allowed_sources"], list) and kwargs["allowed_sources"] and isinstance(kwargs["allowed_sources"][0], dict):
        body["allowed_sources"] = [RetrieveSource(**row) for row in kwargs["allowed_sources"]]
    return RetrieveRequest.model_validate(body)


IDENTITY = fake_embedder_identity()


def _run(req, **kwargs):
    kwargs.setdefault("embed_identity", IDENTITY)
    return run_retrieve(req, **kwargs)


def _row(**kwargs):
    identity = IDENTITY
    row = {
        "id": CHUNK,
        "xa_id": "xa_chu_se",
        "document_id": DOC,
        "generation_id": GEN,
        "procedure_version_id": VER,
        "content": "Bản chính giấy chứng sinh do cơ sở y tế cấp khi đăng ký khai sinh.",
        "page_start": 3,
        "page_end": 3,
        "title": "Hướng dẫn đăng ký khai sinh",
        "document_number": "01/HD",
        "embedding_model_id": identity.model_id,
        "embedding_revision": identity.revision,
        "embedding_checksum": identity.checksum,
    }
    row.update(kwargs)
    if "text_sha256" not in kwargs:
        row["text_sha256"] = hashlib.sha256(str(row.get("content") or "").encode("utf-8")).hexdigest()
    return row


def test_classify_support_fails_closed():
    item = "Bản chính giấy chứng sinh do cơ sở y tế cấp"
    assert classify_support("đăng ký khai sinh tại xã", ["Mang bản chính giấy chứng sinh"]) == "related"
    assert classify_support(item + " nộp tại UBND xã", [item]) == "related"
    assert classify_support("Không cần nộp " + item, [item]) == "related"
    assert classify_support("topic only", []) == "related"


def test_query_embedder_reuses_process_instance():
    reset_query_embedder_cache()
    class Settings:
        embedding_provider = "fake"
        embedding_model_dir = ""

    first, ident_a = query_embedder(Settings())
    second, ident_b = query_embedder(Settings())
    assert first is second
    assert ident_a == ident_b == IDENTITY
    reset_query_embedder_cache()


def test_qdrant_filter_is_prefilter_not_postfilter():
    flt = qdrant_filter(_req())
    keys = {row["key"] for row in flt["must"]}
    assert keys == {"xa_id", "procedure_version_id", "generation_id", "document_id"}
    gen = next(row["match"]["value"] for row in flt["must"] if row["key"] == "generation_id")
    doc = next(row["match"]["value"] for row in flt["must"] if row["key"] == "document_id")
    assert gen == GEN
    assert doc == DOC
    assert all("any" not in (row.get("match") or {}) for row in flt["must"])


def test_empty_allowed_sources_does_not_search():
    called = {"search": False}

    def search(*_args, **_kwargs):
        called["search"] = True
        return []

    resp = _run(
        _req(allowed_sources=[]),
        embed_query=lambda q: [0.1] * E5_DIMENSION,
        search=search,
        load_chunks=lambda _ids: [],
    )
    assert resp.status == STATUS_NONE
    assert called["search"] is False


def test_query_embed_uses_query_prefix():
    fake = FakeEmbedder()
    query_vec = fake.embed_query("khai sinh")
    passage_vec = fake.embed_batch(["khai sinh"])[0]
    assert len(query_vec) == E5_DIMENSION
    assert query_vec != passage_vec
    assert fake.embed_query(QUERY_PREFIX.strip())  # prefix contract stays importable


def test_postgres_mismatch_rejects_stale_qdrant_point():
    def search(_vector, _flt, _k):
        return [
            {
                "score": 0.9,
                "payload": {
                    "chunk_id": CHUNK,
                    "document_id": DOC,
                    "generation_id": GEN,
                    "procedure_version_id": VER,
                    "xa_id": "xa_chu_se",
                },
            }
        ]

    resp = _run(
        _req(),
        embed_query=lambda q: [0.01] * E5_DIMENSION,
        search=search,
        load_chunks=lambda _ids: [],
    )
    assert resp.status == STATUS_REJECTED
    assert resp.hits == []


def test_wrong_commune_and_generation_are_dropped_before_postgres():
    seen_ids = []

    def search(_vector, _flt, _k):
        return [
            {
                "score": 0.99,
                "payload": {
                    "chunk_id": OTHER,
                    "document_id": DOC,
                    "generation_id": OTHER,
                    "procedure_version_id": VER,
                    "xa_id": "xa_other_fixture",
                },
            }
        ]

    def load(ids):
        seen_ids.extend(ids)
        return []

    resp = _run(
        _req(),
        embed_query=lambda q: [0.01] * E5_DIMENSION,
        search=search,
        load_chunks=load,
    )
    assert resp.status == STATUS_REJECTED
    assert seen_ids == []


def test_qdrant_filter_uses_allowed_pairs_not_cartesian():
    other_doc = "66666666-6666-6666-6666-666666666666"
    other_gen = "77777777-7777-7777-7777-777777777777"
    flt = qdrant_filter(
        _req(
            allowed_sources=[
                {"document_id": DOC, "generation_id": GEN},
                {"document_id": other_doc, "generation_id": other_gen},
            ]
        )
    )
    nested = next(row for row in flt["must"] if "should" in row)
    pairs = set()
    for clause in nested["should"]:
        fields = {item["key"]: item["match"]["value"] for item in clause["must"]}
        pairs.add((fields["document_id"], fields["generation_id"]))
    assert pairs == {(DOC, GEN), (other_doc, other_gen)}
    assert all("any" not in (row.get("match") or {}) for row in flt["must"] if "key" in row)


def test_validated_hit_keeps_chunk_identity_and_stays_related():
    excerpt = "Bản chính giấy chứng sinh do cơ sở y tế cấp khi đăng ký khai sinh."

    def search(_vector, _flt, _k):
        return [
            {
                "score": 0.77,
                "payload": {
                    "chunk_id": CHUNK,
                    "document_id": DOC,
                    "generation_id": GEN,
                    "procedure_version_id": VER,
                    "xa_id": "xa_chu_se",
                    "page_start": 3,
                    "page_end": 3,
                },
            }
        ]

    def load(ids):
        assert ids == [CHUNK]
        return [_row(content=excerpt)]

    resp = _run(
        _req(),
        embed_query=lambda q: [0.01] * E5_DIMENSION,
        search=search,
        load_chunks=load,
    )
    assert resp.status == STATUS_OK
    hit = resp.hits[0]
    assert hit.chunk_id == CHUNK
    assert hit.page_start == 3
    assert hit.evidence_status == "related"
    assert hit.title == "Hướng dẫn đăng ký khai sinh"
    assert hit.retrieval_score == pytest.approx(0.77)


def test_page_mismatch_between_qdrant_and_postgres_is_rejected():
    def search(_vector, _flt, _k):
        return [
            {
                "score": 0.9,
                "payload": {
                    "chunk_id": CHUNK,
                    "document_id": DOC,
                    "generation_id": GEN,
                    "procedure_version_id": VER,
                    "xa_id": "xa_chu_se",
                    "page_start": 1,
                    "page_end": 1,
                },
            }
        ]

    resp = _run(
        _req(),
        embed_query=lambda q: [0.01] * E5_DIMENSION,
        search=search,
        load_chunks=lambda _ids: [_row()],
    )
    assert resp.status == STATUS_REJECTED
    assert resp.reason == "postgres_mismatch"


def test_embedding_identity_mismatch_rejects_candidate():
    def search(_vector, _flt, _k):
        return [
            {
                "score": 0.9,
                "payload": {
                    "chunk_id": CHUNK,
                    "document_id": DOC,
                    "generation_id": GEN,
                    "procedure_version_id": VER,
                    "xa_id": "xa_chu_se",
                    "page_start": 3,
                    "page_end": 3,
                },
            }
        ]

    resp = _run(
        _req(),
        embed_query=lambda q: [0.01] * E5_DIMENSION,
        search=search,
        load_chunks=lambda _ids: [_row(embedding_revision="other-revision")],
    )
    assert resp.status == STATUS_INCOMPATIBLE
    assert resp.reason == "embedding_identity_mismatch"
    assert resp.hits == []


def test_missing_embed_identity_is_unavailable():
    with pytest.raises(RetrieveError) as raised:
        run_retrieve(
            _req(),
            embed_query=lambda q: [0.01] * E5_DIMENSION,
            search=lambda *_a, **_k: [],
            load_chunks=lambda _ids: [],
            embed_identity=None,
        )
    assert raised.value.code == "embedding_unavailable"


def test_dimension_mismatch_is_unavailable_contract():
    with pytest.raises(RetrieveError) as raised:
        _run(
            _req(),
            embed_query=lambda q: [0.1] * 8,
            search=lambda *_a, **_k: [],
            load_chunks=lambda _ids: [],
        )
    assert raised.value.code == "dimension_mismatch"


def test_http_empty_scope_and_auth():
    body = {
        "schema_version": "retrieve.v1",
        "query": "khai sinh",
        "xa_id": "xa_chu_se",
        "procedure_version_id": str(uuid.uuid4()),
        "allowed_sources": [],
        "top_k": 3,
        "statements": [],
    }
    denied = client.post("/v1/retrieve", json=body)
    assert denied.status_code == 401
    ok = client.post("/v1/retrieve", json=body, headers=AUTH_HEADER)
    assert ok.status_code == 200
    assert ok.json()["status"] == "no_relevant_evidence"
    assert ok.json()["hits"] == []


def test_http_rejects_too_many_allowed_sources():
    sources = [
        {"document_id": str(uuid.uuid4()), "generation_id": str(uuid.uuid4())}
        for _ in range(MAX_ALLOWED_SOURCES + 1)
    ]
    body = {
        "schema_version": "retrieve.v1",
        "query": "khai sinh",
        "xa_id": "xa_chu_se",
        "procedure_version_id": str(uuid.uuid4()),
        "allowed_sources": sources,
        "top_k": 3,
        "statements": [],
    }
    r = client.post("/v1/retrieve", json=body, headers=AUTH_HEADER)
    assert r.status_code == 422
    assert "traceback" not in str(r.json()).lower()


def test_allowed_sources_at_cap_validates():
    sources = [RetrieveSource(document_id=str(uuid.uuid4()), generation_id=str(uuid.uuid4())) for _ in range(MAX_ALLOWED_SOURCES)]
    req = _req(allowed_sources=sources, query="khai sinh")
    assert len(req.allowed_sources) == MAX_ALLOWED_SOURCES


def test_http_rejects_empty_query():
    body = {
        "schema_version": "retrieve.v1",
        "query": "   ",
        "xa_id": "xa_chu_se",
        "procedure_version_id": str(uuid.uuid4()),
        "allowed_sources": [],
        "top_k": 3,
    }
    r = client.post("/v1/retrieve", json=body, headers=AUTH_HEADER)
    assert r.status_code == 422


def _hit(payload_extra=None, chunk_id=CHUNK):
    payload = {
        "chunk_id": chunk_id,
        "document_id": DOC,
        "generation_id": GEN,
        "procedure_version_id": VER,
        "xa_id": "xa_chu_se",
        "page_start": 3,
        "page_end": 3,
    }
    if payload_extra:
        payload.update(payload_extra)
    return {"score": 0.8, "payload": payload}


def test_malformed_page_start_rejects_candidate_not_crash():
    resp = _run(
        _req(),
        embed_query=lambda q: [0.01] * E5_DIMENSION,
        search=lambda *_a, **_k: [_hit({"page_start": "oops"})],
        load_chunks=lambda _ids: [_row()],
    )
    assert resp.status == STATUS_REJECTED
    assert resp.hits == []


def test_missing_and_null_page_fields_reject_candidate():
    resp = _run(
        _req(),
        embed_query=lambda q: [0.01] * E5_DIMENSION,
        search=lambda *_a, **_k: [
            {
                "score": 0.8,
                "payload": {
                    "chunk_id": CHUNK,
                    "document_id": DOC,
                    "generation_id": GEN,
                    "procedure_version_id": VER,
                    "xa_id": "xa_chu_se",
                },
            }
        ],
        load_chunks=lambda _ids: [_row(page_start=None, page_end=None)],
    )
    assert resp.status == STATUS_REJECTED
    assert resp.reason == "postgres_mismatch"
    assert resp.hits == []


def test_inverted_page_range_rejects_candidate():
    assert valid_page_range(None, 3) is False
    assert valid_page_range(0, 3) is False
    assert valid_page_range(12, 3) is False
    assert valid_page_range(3, 3) is True
    resp = _run(
        _req(),
        embed_query=lambda q: [0.01] * E5_DIMENSION,
        search=lambda *_a, **_k: [_hit({"page_start": 12, "page_end": 3})],
        load_chunks=lambda _ids: [_row(page_start=12, page_end=3)],
    )
    assert resp.status == STATUS_REJECTED
    assert resp.hits == []


def test_content_hash_mismatch_rejects_candidate():
    resp = _run(
        _req(),
        embed_query=lambda q: [0.01] * E5_DIMENSION,
        search=lambda *_a, **_k: [_hit()],
        load_chunks=lambda _ids: [_row(text_sha256="b" * 64)],
    )
    assert resp.status == STATUS_REJECTED
    assert resp.reason == "postgres_mismatch"
    assert resp.hits == []


def test_printer_text_above_floor_is_not_related():
    printer = "Báo cáo lỗi máy in văn phòng tháng 10."
    assert has_query_overlap("Tôi muốn đăng ký khai sinh", printer) is False
    resp = _run(
        _req(),
        embed_query=lambda q: [0.01] * E5_DIMENSION,
        search=lambda *_a, **_k: [_hit() | {"score": 0.11}],
        load_chunks=lambda _ids: [_row(content=printer)],
    )
    assert resp.status == STATUS_NONE
    assert resp.reason == "no_query_overlap"
    assert resp.hits == []
    assert classify_support(printer, ["Bản chính giấy chứng sinh do cơ sở y tế cấp"]) == "related"


def test_conversational_printer_is_not_related_when_checklist_present():
    printer = "Tôi muốn xem thông tin sửa chữa máy in văn phòng"
    query = "Tôi muốn đăng ký khai sinh"
    checklist = ["Bản chính giấy chứng sinh do cơ sở y tế cấp"]
    assert has_query_overlap(query, printer) is True
    assert has_related_overlap(query, printer, checklist) is False
    resp = _run(
        _req(),
        embed_query=lambda q: [0.01] * E5_DIMENSION,
        search=lambda *_a, **_k: [_hit() | {"score": 0.11}],
        load_chunks=lambda _ids: [_row(content=printer)],
    )
    assert resp.status == STATUS_NONE
    assert resp.reason == "no_query_overlap"
    assert resp.hits == []


def test_short_slot_answer_keeps_checklist_overlap():
    excerpt = "Bản chính giấy chứng sinh do cơ sở y tế cấp khi đăng ký khai sinh."
    checklist = ["Bản chính giấy chứng sinh do cơ sở y tế cấp"]
    assert has_related_overlap("Dạ có", excerpt, checklist) is True
    resp = _run(
        _req(query="Dạ có"),
        embed_query=lambda q: [0.01] * E5_DIMENSION,
        search=lambda *_a, **_k: [_hit() | {"score": 0.77}],
        load_chunks=lambda _ids: [_row(content=excerpt)],
    )
    assert resp.status == STATUS_OK
    assert len(resp.hits) == 1
    assert resp.hits[0].evidence_status == "related"


def test_g3_identity_without_allowlist_stays_related():
    birth = "Giấy chứng sinh (nếu có) hoặc giấy tờ thay thế theo hướng dẫn xã"
    parent_id = "CCCD/hộ chiếu của cha và/hoặc mẹ"
    marriage = "Giấy đăng ký kết hôn (nếu cha mẹ đã kết hôn)"
    form = "Tờ khai đăng ký khai sinh (mẫu tại bộ phận tiếp nhận)"
    g3 = dict(
        document_id="60b677df-acc0-4198-a3b6-a919c01482a1",
        generation_id="e2ad49da-4db7-4e18-90e3-c2f0bb358940",
        procedure_code="dk_khai_sinh",
    )
    assert (
        map_classify_support(
            "overlap is not enough",
            [birth],
            page_start=12,
            page_end=12,
            text_sha256="410276e9993151a4f2ff17e5e8a3ea7b5cdc6abd42faf6ded0ac07cf57e8b17c",
            **g3,
        )
        == "related"
    )
    assert (
        map_classify_support(
            "tờ khai",
            [form],
            page_start=12,
            page_end=12,
            text_sha256="f355aaa615088e28a895b5ac3f26d10447db6d6ffdf95cb1db8e1a6f1e0e126a",
            **g3,
        )
        == "related"
    )
    assert (
        map_classify_support(
            "ủy quyền thực hiện việc đăng ký khai sinh; giấy tờ tùy thân",
            [parent_id],
            page_start=13,
            page_end=13,
            text_sha256="dbb5a7a599f32d2d5ec7be05f210eccf8dcb1bb1216d8c297816c026049887ef",
            **g3,
        )
        == "related"
    )
    assert (
        map_classify_support(
            "kết hôn",
            [marriage],
            page_start=12,
            page_end=12,
            text_sha256="410276e9993151a4f2ff17e5e8a3ea7b5cdc6abd42faf6ded0ac07cf57e8b17c",
            **g3,
        )
        == "related"
    )
    assert (
        map_classify_support(
            "Giấy chứng sinh",
            [birth],
            document_id="60b677df-acc0-4198-a3b6-a919c01482a1",
            generation_id="06679d02-3a43-4831-8dde-82799fc9b4a4",
            page_start=12,
            page_end=12,
            text_sha256="410276e9993151a4f2ff17e5e8a3ea7b5cdc6abd42faf6ded0ac07cf57e8b17c",
            procedure_code="dk_khai_sinh",
        )
        == "related"
    )


def test_g3_gcs_entailed_statement_is_supporting():
    statement = (
        "Giấy chứng sinh; nếu không có Giấy chứng sinh thì văn bản người làm chứng "
        "xác nhận việc sinh; nếu không có người làm chứng thì giấy cam đoan về việc sinh"
    )
    common = dict(
        document_id="60b677df-acc0-4198-a3b6-a919c01482a1",
        page_start=12,
        page_end=12,
        text_sha256="410276e9993151a4f2ff17e5e8a3ea7b5cdc6abd42faf6ded0ac07cf57e8b17c",
        procedure_code="dk_khai_sinh",
    )
    assert (
        map_classify_support(
            "identity only",
            [statement],
            generation_id="e2ad49da-4db7-4e18-90e3-c2f0bb358940",
            **common,
        )
        == "supporting"
    )
    assert (
        map_classify_support(
            "identity only",
            [statement],
            generation_id="21125806-0a32-4e84-a866-d497859be2fe",
            **common,
        )
        == "supporting"
    )


def test_khai_sinh_excerpt_above_floor_stays_related_not_supporting():
    excerpt = "Bản chính giấy chứng sinh do cơ sở y tế cấp khi đăng ký khai sinh."
    assert has_query_overlap("Tôi muốn đăng ký khai sinh", excerpt) is True
    resp = _run(
        _req(),
        embed_query=lambda q: [0.01] * E5_DIMENSION,
        search=lambda *_a, **_k: [_hit() | {"score": 0.77}],
        load_chunks=lambda _ids: [_row(content=excerpt)],
    )
    assert resp.status == STATUS_OK
    assert len(resp.hits) == 1
    assert resp.hits[0].evidence_status == "related"
    assert resp.hits[0].retrieval_score == 0.77


def test_mixed_valid_and_malformed_candidates_keep_valid():
    other = "88888888-8888-8888-8888-888888888888"

    def search(_v, _f, _k):
        return [_hit({"page_start": "oops"}), _hit({"chunk_id": other})]

    def load(ids):
        return [_row(), _row(id=other, content="Đăng ký khai sinh tại UBND xã.")]

    resp = _run(
        _req(),
        embed_query=lambda q: [0.01] * E5_DIMENSION,
        search=search,
        load_chunks=load,
    )
    assert resp.status == STATUS_OK
    assert [hit.chunk_id for hit in resp.hits] == [other]


def test_no_candidates_is_no_relevant_evidence():
    resp = _run(
        _req(),
        embed_query=lambda q: [0.01] * E5_DIMENSION,
        search=lambda *_a, **_k: [],
        load_chunks=lambda _ids: [],
    )
    assert resp.status == STATUS_NONE
    assert resp.hits == []


def test_qdrant_unavailable_is_retrieve_error():
    def search(*_a, **_k):
        raise RetrieveError("qdrant_unavailable")

    with pytest.raises(RetrieveError) as raised:
        _run(_req(), embed_query=lambda q: [0.01] * E5_DIMENSION, search=search, load_chunks=lambda _ids: [])
    assert raised.value.code == "qdrant_unavailable"


def test_http_embedder_init_failure_is_structured(monkeypatch):
    from app.indexing import retrieve as retrieve_mod

    def boom(_settings):
        raise RetrieveError("embedding_unavailable")

    monkeypatch.setattr(retrieve_mod, "query_embedder", boom)
    body = {
        "schema_version": "retrieve.v1",
        "query": "khai sinh",
        "xa_id": "xa_chu_se",
        "procedure_version_id": VER,
        "allowed_sources": [{"document_id": DOC, "generation_id": GEN}],
        "top_k": 3,
        "statements": [],
    }
    r = client.post("/v1/retrieve", json=body, headers=AUTH_HEADER)
    assert r.status_code == 503
    assert r.json()["detail"]["reason"] == "embedding_unavailable"
    assert "traceback" not in str(r.json()).lower()
    assert "/models" not in str(r.json())


def test_onnx_missing_directory_raises_unavailable():
    reset_query_embedder_cache()

    class Settings:
        embedding_provider = "onnx"
        embedding_model_dir = "/tmp/cas-missing-onnx-dir"

    with pytest.raises(RetrieveError) as raised:
        query_embedder(Settings())
    assert raised.value.code == "embedding_unavailable"
    reset_query_embedder_cache()


def test_query_embedder_concurrent_reuses_one_instance():
    reset_query_embedder_cache()
    import threading

    class Settings:
        embedding_provider = "fake"
        embedding_model_dir = ""

    got = []

    def worker():
        fn, _ident = query_embedder(Settings())
        got.append(fn)

    threads = [threading.Thread(target=worker) for _ in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(got) == 12
    assert len({id(fn) for fn in got}) == 1
    reset_query_embedder_cache()


def test_related_floor_rejects_noise_cosine_scores():
    assert passes_related_floor(-0.80) is False
    assert passes_related_floor(0.0) is False
    assert passes_related_floor(0.01) is False
    assert passes_related_floor(0.10) is True
    assert passes_related_floor(0.77) is True
    assert passes_related_floor(None) is False
    assert passes_related_floor(True) is False

    def search(_v, _f, _k):
        return [_hit() | {"score": -0.80}]

    resp = _run(_req(), embed_query=lambda q: [0.01] * E5_DIMENSION, search=search, load_chunks=lambda _ids: [_row()])
    assert resp.status == STATUS_NONE
    assert resp.reason == "below_relevance_floor"
    assert resp.hits == []


@pytest.mark.parametrize("score", [0.00, 0.01])
def test_near_zero_cosine_is_not_related(score):
    resp = _run(
        _req(),
        embed_query=lambda q: [0.01] * E5_DIMENSION,
        search=lambda *_a, **_k: [_hit() | {"score": score}],
        load_chunks=lambda _ids: [_row()],
    )
    assert resp.status == STATUS_NONE
    assert resp.hits == []


def test_malformed_payload_object_rejects_candidate_not_request():
    other = "88888888-8888-8888-8888-888888888888"

    def search(_v, _f, _k):
        return [
            {"score": 0.8, "payload": "oops"},
            {"score": 0.8, "payload": 12},
            "not-a-hit",
            _hit({"chunk_id": other}),
        ]

    def load(ids):
        return [_row(), _row(id=other, content="Đăng ký khai sinh tại UBND xã.")]

    resp = _run(_req(), embed_query=lambda q: [0.01] * E5_DIMENSION, search=search, load_chunks=load)
    assert resp.status == STATUS_OK
    assert [hit.chunk_id for hit in resp.hits] == [other]


def test_all_malformed_payloads_do_not_crash():
    resp = _run(
        _req(),
        embed_query=lambda q: [0.01] * E5_DIMENSION,
        search=lambda *_a, **_k: [{"score": 0.9, "payload": "oops"}, {"score": 0.8, "payload": 1}],
        load_chunks=lambda _ids: [_row()],
    )
    assert resp.status == STATUS_REJECTED
    assert resp.hits == []
