"""Scoped Qdrant search + PostgreSQL authoritative validation."""

from __future__ import annotations

import hashlib
import math
import threading
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable

import httpx

from app.indexing.embed import E5_DIMENSION, FAKE_MODEL_ID, FAKE_REVISION, fake_checksum, normalize
from app.indexing.pipeline import FakeEmbedder
from app.indexing.support import classify_support
from app.indexing.text import sha256_text
from app.models.retrieve import RetrieveHit, RetrieveRequest, RetrieveResponse

STATUS_OK = "ok"
STATUS_NONE = "no_relevant_evidence"
STATUS_REJECTED = "all_candidates_rejected"
STATUS_UNAVAILABLE = "retrieval_unavailable"
STATUS_INCOMPATIBLE = "retrieval_incompatible"

# Noise floor for L2-normalized cosine. Drops anti-aligned and near-orthogonal
# Qdrant scores so "related" is not applied to -0.80 / 0.00 / 0.01 noise.
# This is not a calibrated legal-relevance threshold and not supporting proof.
MIN_RELATED_COSINE = 0.10

_query_embedders: dict[tuple[str, str], tuple[Callable[[str], list[float]], "EmbedderIdentity"]] = {}
_query_embedders_lock = threading.Lock()


class RetrieveError(Exception):
    def __init__(self, code: str, http_status: int = 503) -> None:
        self.code = code
        self.http_status = http_status
        super().__init__(code)


@dataclass(frozen=True)
class EmbedderIdentity:
    model_id: str
    revision: str
    checksum: str


def fake_embedder_identity() -> EmbedderIdentity:
    return EmbedderIdentity(model_id=FAKE_MODEL_ID, revision=FAKE_REVISION, checksum=fake_checksum())


def passes_related_floor(score: Any) -> bool:
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return False
    value = float(score)
    return math.isfinite(value) and value >= MIN_RELATED_COSINE


def query_tokens(text: str) -> list[str]:
    """Same rule as Go chat.tokens: lowercase, split on space/punct, drop 1-char tokens."""
    out: list[str] = []
    buf: list[str] = []
    for ch in str(text or "").lower():
        if ch.isspace() or unicodedata.category(ch).startswith("P"):
            if buf:
                token = "".join(buf)
                if len(token) >= 2:
                    out.append(token)
                buf = []
            continue
        buf.append(ch)
    if buf:
        token = "".join(buf)
        if len(token) >= 2:
            out.append(token)
    return out


def has_query_overlap(query: str, excerpt: str) -> bool:
    """Relatedness gate: at least one query token must appear in the excerpt.

    This is the same fail-closed drop as lexical Score<=0. It is not supporting
    proof and not a calibrated legal cosine threshold.
    """
    terms = query_tokens(query)
    if not terms:
        return False
    folded = str(excerpt or "").lower()
    return any(term in folded for term in terms)


def relatedness_texts(query: str, statements: list[str] | None) -> list[str]:
    """Prefer Decision Engine checklist language over the latest utterance.

    Short slot answers ("Dạ có") must not drop pinned PDFs. Conversational
    prefixes shared with the question ("Tôi muốn") are not relatedness proof.
    """
    texts = [" ".join(str(item or "").split()) for item in (statements or [])]
    texts = [item for item in texts if item]
    if texts:
        return texts
    q = str(query or "").strip()
    return [q] if q else []


def has_related_overlap(query: str, excerpt: str, statements: list[str] | None = None) -> bool:
    return any(has_query_overlap(text, excerpt) for text in relatedness_texts(query, statements))


def valid_page_range(page_start: int | None, page_end: int | None) -> bool:
    return page_start is not None and page_end is not None and page_start >= 1 and page_end >= page_start


def hit_payload(hit: Any) -> dict[str, Any] | None:
    if not isinstance(hit, dict):
        return None
    payload = hit.get("payload")
    if not isinstance(payload, dict):
        return None
    return payload


def allowed_pairs(req: RetrieveRequest) -> set[tuple[str, str]]:
    return {(row.document_id, row.generation_id) for row in req.allowed_sources}


def qdrant_filter(req: RetrieveRequest) -> dict[str, Any]:
    must: list[dict[str, Any]] = [
        {"key": "xa_id", "match": {"value": req.xa_id}},
        {"key": "procedure_version_id", "match": {"value": req.procedure_version_id}},
    ]
    pairs = sorted(allowed_pairs(req))
    if len(pairs) == 1:
        document_id, generation_id = pairs[0]
        must.append({"key": "document_id", "match": {"value": document_id}})
        must.append({"key": "generation_id", "match": {"value": generation_id}})
    elif pairs:
        must.append(
            {
                "should": [
                    {
                        "must": [
                            {"key": "document_id", "match": {"value": document_id}},
                            {"key": "generation_id", "match": {"value": generation_id}},
                        ]
                    }
                    for document_id, generation_id in pairs
                ]
            }
        )
    return {"must": must}


def run_retrieve(
    req: RetrieveRequest,
    *,
    embed_query: Callable[[str], list[float]],
    search: Callable[[list[float], dict[str, Any], int], list[dict[str, Any]]],
    load_chunks: Callable[[list[str]], list[dict[str, Any]]],
    embed_identity: EmbedderIdentity | None = None,
) -> RetrieveResponse:
    if not req.allowed_sources:
        return RetrieveResponse(status=STATUS_NONE, reason="empty_allowed_sources")
    if embed_identity is None or not embed_identity.model_id or not embed_identity.revision:
        raise RetrieveError("embedding_unavailable")
    try:
        vector = embed_query(req.query)
    except RetrieveError:
        raise
    except Exception as exc:
        raise RetrieveError("embedding_unavailable") from exc
    if len(vector) != E5_DIMENSION or any(not math.isfinite(x) for x in vector):
        raise RetrieveError("dimension_mismatch", http_status=422)
    vector = normalize(vector)
    try:
        raw_hits = search(vector, qdrant_filter(req), req.top_k)
    except RetrieveError:
        raise
    except Exception as exc:
        raise RetrieveError("qdrant_unavailable") from exc
    if not raw_hits:
        return RetrieveResponse(status=STATUS_NONE, reason="no_candidates")
    allowed = allowed_pairs(req)
    scoped: list[dict[str, Any]] = []
    chunk_ids: list[str] = []
    for hit in raw_hits:
        payload = hit_payload(hit)
        if payload is None:
            continue
        chunk_id = str(payload.get("chunk_id") or "")
        document_id = str(payload.get("document_id") or "")
        generation_id = str(payload.get("generation_id") or "")
        version_id = str(payload.get("procedure_version_id") or "")
        xa_id = str(payload.get("xa_id") or "")
        if not chunk_id or (document_id, generation_id) not in allowed:
            continue
        if xa_id != req.xa_id or version_id != req.procedure_version_id:
            continue
        scoped.append(hit)
        chunk_ids.append(chunk_id)
    if not chunk_ids:
        return RetrieveResponse(status=STATUS_REJECTED, reason="filter_mismatch")
    try:
        rows = load_chunks(chunk_ids)
    except RetrieveError:
        raise
    except Exception as exc:
        raise RetrieveError("postgres_unavailable") from exc
    by_id = {str(row.get("id")): row for row in rows}
    hits: list[RetrieveHit] = []
    identity_mismatch = False
    metadata_mismatch = False
    below_floor = False
    no_overlap = False
    for hit in scoped:
        payload = hit_payload(hit)
        if payload is None:
            metadata_mismatch = True
            continue
        chunk_id = str(payload.get("chunk_id") or "")
        row = by_id.get(chunk_id)
        if row is None:
            metadata_mismatch = True
            continue
        if not _payload_matches_row(payload, row, req, allowed):
            metadata_mismatch = True
            continue
        if not _identity_matches(embed_identity, row):
            identity_mismatch = True
            continue
        raw = str(row.get("content") or "")
        excerpt = raw.strip()
        if not excerpt:
            continue
        stored_hash = str(row.get("text_sha256") or "")
        if stored_hash != sha256_text(raw):
            metadata_mismatch = True
            continue
        page_start, page_ok = _optional_int(row.get("page_start"))
        page_end, page_end_ok = _optional_int(row.get("page_end"))
        if not page_ok or not page_end_ok or not valid_page_range(page_start, page_end):
            metadata_mismatch = True
            continue
        assert page_start is not None and page_end is not None
        score = hit.get("score")
        if not passes_related_floor(score):
            below_floor = True
            continue
        if not has_related_overlap(req.query, excerpt, req.statements):
            no_overlap = True
            continue
        retrieval_score = float(score)
        hits.append(
            RetrieveHit(
                chunk_id=chunk_id,
                document_id=str(row.get("document_id") or ""),
                generation_id=str(row.get("generation_id") or ""),
                procedure_version_id=str(row.get("procedure_version_id") or req.procedure_version_id),
                title=str(row.get("title") or ""),
                document_number=str(row.get("document_number") or ""),
                page_start=page_start,
                page_end=page_end,
                excerpt=excerpt,
                excerpt_sha256=hashlib.sha256(excerpt.encode("utf-8")).hexdigest(),
                content_sha256=stored_hash,
                retrieval_score=retrieval_score,
                evidence_status=classify_support(
                    excerpt,
                    req.statements,
                    document_id=str(row.get("document_id") or ""),
                    generation_id=str(row.get("generation_id") or ""),
                    page_start=page_start,
                    page_end=page_end,
                    text_sha256=stored_hash,
                ),
                availability="available",
            )
        )
    if not hits:
        if identity_mismatch and not metadata_mismatch and not below_floor and not no_overlap:
            return RetrieveResponse(status=STATUS_INCOMPATIBLE, reason="embedding_identity_mismatch")
        if below_floor and not metadata_mismatch and not identity_mismatch:
            return RetrieveResponse(status=STATUS_NONE, reason="below_relevance_floor")
        if no_overlap and not metadata_mismatch and not identity_mismatch:
            return RetrieveResponse(status=STATUS_NONE, reason="no_query_overlap")
        return RetrieveResponse(status=STATUS_REJECTED, reason="postgres_mismatch")
    return RetrieveResponse(status=STATUS_OK, hits=hits)


def _payload_matches_row(payload: dict[str, Any], row: dict[str, Any], req: RetrieveRequest, allowed: set[tuple[str, str]]) -> bool:
    document_id = str(row.get("document_id") or "")
    generation_id = str(row.get("generation_id") or "")
    if str(row.get("xa_id") or "") != req.xa_id:
        return False
    if str(row.get("procedure_version_id") or "") != req.procedure_version_id:
        return False
    if (document_id, generation_id) not in allowed:
        return False
    for key in ("xa_id", "document_id", "generation_id", "procedure_version_id"):
        if str(payload.get(key) or "") != str(row.get(key) or ""):
            return False
    if "page_start" in payload:
        want, ok = _optional_int(payload.get("page_start"))
        got, got_ok = _optional_int(row.get("page_start"))
        if not ok or not got_ok or want is None or got is None or want != got:
            return False
    if "page_end" in payload:
        want, ok = _optional_int(payload.get("page_end"))
        got, got_ok = _optional_int(row.get("page_end"))
        if not ok or not got_ok or want is None or got is None or want != got:
            return False
    return True


def _optional_int(value: Any) -> tuple[int | None, bool]:
    if value is None or value == "":
        return None, True
    if isinstance(value, bool):
        return None, False
    if isinstance(value, int):
        return value, True
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return int(value), True
        return None, False
    text = str(value).strip()
    if text.lstrip("-").isdigit():
        return int(text), True
    return None, False


def _identity_matches(identity: EmbedderIdentity, row: dict[str, Any]) -> bool:
    return (
        str(row.get("embedding_model_id") or "") == identity.model_id
        and str(row.get("embedding_revision") or "") == identity.revision
        and str(row.get("embedding_checksum") or "") == identity.checksum
    )


def http_search(base_url: str, collection: str, timeout_s: float) -> Callable[[list[float], dict[str, Any], int], list[dict[str, Any]]]:
    def search(vector: list[float], flt: dict[str, Any], limit: int) -> list[dict[str, Any]]:
        if not base_url or not collection:
            raise RetrieveError("qdrant_unavailable")
        url = f"{base_url.rstrip('/')}/collections/{collection}/points/search"
        body = {"vector": vector, "limit": limit, "with_payload": True, "filter": flt}
        try:
            response = httpx.post(url, json=body, timeout=timeout_s)
        except httpx.TimeoutException as exc:
            raise RetrieveError("timeout", http_status=504) from exc
        except httpx.HTTPError as exc:
            raise RetrieveError("qdrant_unavailable") from exc
        if response.status_code != 200:
            raise RetrieveError("qdrant_unavailable")
        payload = response.json()
        result = payload.get("result") if isinstance(payload, dict) else None
        if not isinstance(result, list):
            raise RetrieveError("qdrant_unavailable")
        return [row for row in result if isinstance(row, dict)]

    return search


def postgres_loader(dsn: str, timeout_s: float) -> Callable[[list[str]], list[dict[str, Any]]]:
    def load_chunks(chunk_ids: list[str]) -> list[dict[str, Any]]:
        if not dsn:
            raise RetrieveError("postgres_unavailable")
        try:
            import psycopg
        except ImportError as exc:
            raise RetrieveError("postgres_unavailable") from exc
        sql = """
            SELECT c.id::text AS id,
                   c.xa_id,
                   c.document_id::text AS document_id,
                   c.generation_id::text AS generation_id,
                   c.procedure_version_id::text AS procedure_version_id,
                   c.content,
                   c.page_start,
                   c.page_end,
                   c.text_sha256,
                   d.title,
                   COALESCE(d.document_number, '') AS document_number,
                   COALESCE(g.content_sha256, '') AS content_sha256,
                   COALESCE(g.embedding_model_id, '') AS embedding_model_id,
                   COALESCE(g.embedding_revision, '') AS embedding_revision,
                   COALESCE(g.embedding_checksum, '') AS embedding_checksum
            FROM knowledge_chunks c
            JOIN documents d ON d.id = c.document_id
            JOIN document_index_generations g ON g.id = c.generation_id
            WHERE c.id = ANY(%s::uuid[])
        """
        try:
            with psycopg.connect(dsn, connect_timeout=min(2, int(timeout_s) or 2)) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (chunk_ids,))
                    cols = [c.name for c in cur.description]
                    return [dict(zip(cols, row)) for row in cur.fetchall()]
        except Exception as exc:
            raise RetrieveError("postgres_unavailable") from exc

    return load_chunks


def reset_query_embedder_cache() -> None:
    with _query_embedders_lock:
        _query_embedders.clear()


def query_embedder(settings) -> tuple[Callable[[str], list[float]], EmbedderIdentity]:
    provider = str(getattr(settings, "embedding_provider", "fake") or "fake").strip().lower()
    model_dir = str(getattr(settings, "embedding_model_dir", "") or "")
    key = (provider, model_dir)
    with _query_embedders_lock:
        cached = _query_embedders.get(key)
        if cached is not None:
            return cached
        embed, identity = _build_query_embedder(provider, model_dir)
        _query_embedders[key] = (embed, identity)
        return embed, identity


def _build_query_embedder(provider: str, model_dir: str) -> tuple[Callable[[str], list[float]], EmbedderIdentity]:
    if provider in ("", "fake"):
        fake = FakeEmbedder()
        identity = fake_embedder_identity()

        def embed(text: str) -> list[float]:
            return fake.embed_query(text)

        return embed, identity
    from app.indexing.embed import OnnxEmbedder

    if not model_dir:
        raise RetrieveError("embedding_unavailable")
    try:
        embedder = OnnxEmbedder(model_dir)
    except Exception as exc:
        raise RetrieveError("embedding_unavailable") from exc
    identity = EmbedderIdentity(model_id=embedder.model_id, revision=embedder.revision, checksum=str(embedder.checksum or ""))

    def embed(text: str) -> list[float]:
        return embedder.embed_query(text)

    return embed, identity
