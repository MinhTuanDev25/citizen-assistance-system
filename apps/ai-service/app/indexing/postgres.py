"""Persist staging chunks. Publishing READY stays in the Go API."""

from __future__ import annotations

import json

from app.indexing.deadline import Deadline
from app.indexing.errors import IndexFailure
from app.indexing.pipeline import IndexJob, SavedChunk


class PostgresChunks:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    def replace(self, job: IndexJob, chunks: list[SavedChunk], meta: dict, timeout_s: float | None = None, deadline=None) -> None:
        clock = deadline if deadline is not None else (Deadline(timeout_s) if timeout_s is not None else None)
        try:
            import psycopg
        except ImportError as exc:
            raise IndexFailure("postgres_failed") from exc
        connect_timeout = 2
        if clock is not None:
            left = clock.remaining()
            if left < 1:
                raise IndexFailure("timeout")
            connect_timeout = min(2, int(left))
        try:
            with psycopg.connect(self.dsn, connect_timeout=connect_timeout) as conn:
                close = getattr(conn, "close", None)
                bind = getattr(clock, "bind", None) if clock is not None else None
                if bind is not None and close is not None:
                    bind(close)
                try:
                    with conn.cursor() as cur:
                        self._exec(
                            cur,
                            clock,
                            """
                            UPDATE document_index_generations
                            SET pipeline_version = %s,
                                extraction_version = %s,
                                ocr_version = %s,
                                chunk_config_hash = %s,
                                embedding_model_id = %s,
                                embedding_revision = %s,
                                embedding_checksum = %s,
                                vector_dimension = %s,
                                source_sha256 = %s,
                                content_sha256 = %s,
                                manifest_hash = %s,
                                page_count = %s,
                                native_page_count = %s,
                                ocr_page_count = %s,
                                chunk_count = %s,
                                vector_count = %s,
                                updated_at = now()
                            WHERE id = %s AND xa_id = %s AND document_id = %s
                              AND procedure_version_id = %s AND job_id = %s
                              AND status = 'STAGING'
                            """,
                            (
                                meta["pipeline_version"],
                                meta["extraction_version"],
                                meta["ocr_version"],
                                meta["chunk_config_hash"],
                                meta["embedding_model_id"],
                                meta["embedding_revision"],
                                meta["embedding_checksum"],
                                meta["vector_dimension"],
                                meta["source_sha256"],
                                meta["content_sha256"],
                                meta["manifest_hash"],
                                meta["page_count"],
                                meta["native_page_count"],
                                meta["ocr_page_count"],
                                meta["chunk_count"],
                                meta["vector_count"],
                                job.generation_id,
                                job.xa_id,
                                job.document_id,
                                job.procedure_version_id,
                                job.job_id,
                            ),
                        )
                        if cur.rowcount != 1:
                            raise IndexFailure("claim_rejected")
                        self._exec(
                            cur,
                            clock,
                            "DELETE FROM knowledge_chunks WHERE generation_id = %s AND xa_id = %s",
                            (job.generation_id, job.xa_id),
                        )
                        for chunk in chunks:
                            self._exec(
                                cur,
                                clock,
                                """
                                INSERT INTO knowledge_chunks (
                                    id, xa_id, procedure_id, procedure_version_id, document_id,
                                    chunk_index, content, metadata, embedding, generation_id,
                                    page_start, page_end, text_sha256, token_count, extraction_source
                                ) VALUES (
                                    %s, %s, %s, %s, %s,
                                    %s, %s, %s::jsonb, NULL, %s,
                                    %s, %s, %s, %s, %s
                                )
                                """,
                                (
                                    chunk.chunk_id,
                                    job.xa_id,
                                    job.procedure_id,
                                    job.procedure_version_id,
                                    job.document_id,
                                    chunk.index,
                                    chunk.text,
                                    json.dumps({"source": chunk.source}),
                                    job.generation_id,
                                    chunk.page_start,
                                    chunk.page_end,
                                    chunk.text_sha256,
                                    chunk.token_count,
                                    chunk.source,
                                ),
                            )
                    if clock is not None:
                        clock.check()
                    conn.commit()
                except Exception:
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                    if clock is not None and getattr(clock, "cancelled", False):
                        raise IndexFailure("timeout")
                    raise
                finally:
                    unbind = getattr(clock, "unbind", None) if clock is not None else None
                    if unbind is not None and close is not None:
                        unbind(close)
        except IndexFailure:
            raise
        except Exception as exc:
            raise IndexFailure("postgres_failed") from exc

    def _exec(self, cur, deadline, sql: str, params) -> None:
        if deadline is not None:
            deadline.check()
            cur.execute("SELECT set_config('statement_timeout', %s, true)", (str(max(1, int(deadline.remaining() * 1000))),))
            deadline.check()
        cur.execute(sql, params)
        if deadline is not None:
            deadline.check()
