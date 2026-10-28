"""Job progress and the per-page extraction cache. Both are guarded by the job's claim."""

from __future__ import annotations

import json

from app.indexing.errors import IndexFailure
from app.indexing.pipeline import CachedPage, IndexJob

_PROGRESS_COLUMNS = {
    "stage": "progress_stage",
    "pages_total": "pages_total",
    "pages_done": "pages_done",
    "pages_ocr": "pages_ocr",
    "pages_rejected": "pages_rejected",
    "report": "report",
}


class _Conn:
    """One lazily opened autocommit connection per pipeline run."""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._conn = None

    def cursor(self):
        if self._conn is None or self._conn.closed:
            import psycopg

            self._conn = psycopg.connect(self.dsn, connect_timeout=2, autocommit=True)
        return self._conn.cursor()

    def close(self) -> None:
        if self._conn is not None and not self._conn.closed:
            self._conn.close()


class PostgresProgress:
    def __init__(self, dsn: str) -> None:
        self._db = _Conn(dsn)

    def update(self, job: IndexJob, **fields) -> None:
        sets: list[str] = []
        values: list = []
        for key, value in fields.items():
            column = _PROGRESS_COLUMNS.get(key)
            if column is None:
                raise ValueError(f"unknown progress field {key}")
            if column == "report":
                sets.append("report = %s::jsonb")
                values.append(json.dumps(value))
            else:
                sets.append(f"{column} = %s")
                values.append(value)
        if not sets:
            return
        with self._db.cursor() as cur:
            cur.execute(
                f"""
                UPDATE document_index_jobs
                SET {", ".join(sets)}, progress_updated_at = now()
                WHERE id = %s AND claim_token = %s AND xa_id = %s AND status = 'CLAIMED'
                """,
                (*values, job.job_id, job.claim_token, job.xa_id),
            )
            if cur.rowcount != 1:
                raise IndexFailure("claim_rejected")

    def close(self) -> None:
        self._db.close()


class PostgresPageCache:
    def __init__(self, dsn: str) -> None:
        self._db = _Conn(dsn)

    def load(self, job: IndexJob, source_sha256: str, key: str) -> dict[int, CachedPage]:
        with self._db.cursor() as cur:
            cur.execute(
                """
                SELECT page_number, source, accepted, reason, text, quality
                FROM document_page_extractions
                WHERE document_id = %s AND xa_id = %s AND source_sha256 = %s AND extraction_key = %s
                """,
                (job.document_id, job.xa_id, source_sha256, key),
            )
            rows = cur.fetchall()
        out: dict[int, CachedPage] = {}
        for number, source, accepted, reason, text, quality in rows:
            out[int(number)] = CachedPage(int(number), source, bool(accepted), reason, text or "", quality or {})
        return out

    def store(self, job: IndexJob, source_sha256: str, key: str, page: CachedPage) -> None:
        with self._db.cursor() as cur:
            cur.execute(
                """
                INSERT INTO document_page_extractions (
                    document_id, xa_id, source_sha256, extraction_key, page_number,
                    source, accepted, reason, text, quality
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (document_id, source_sha256, extraction_key, page_number) DO UPDATE
                SET source = EXCLUDED.source, accepted = EXCLUDED.accepted, reason = EXCLUDED.reason,
                    text = EXCLUDED.text, quality = EXCLUDED.quality, created_at = now()
                """,
                (
                    job.document_id, job.xa_id, source_sha256, key, page.number,
                    page.source, page.accepted, page.reason, page.text, json.dumps(page.quality),
                ),
            )

    def close(self) -> None:
        self._db.close()
