-- 000018_phase5_retrieve.down.sql

ALTER TABLE message_citations
    DROP CONSTRAINT IF EXISTS message_citations_evidence_status_check,
    DROP CONSTRAINT IF EXISTS message_citations_availability_check,
    DROP CONSTRAINT IF EXISTS message_citations_generation_fk;

ALTER TABLE message_citations
    DROP COLUMN IF EXISTS excerpt,
    DROP COLUMN IF EXISTS excerpt_sha256,
    DROP COLUMN IF EXISTS generation_id,
    DROP COLUMN IF EXISTS procedure_version_id,
    DROP COLUMN IF EXISTS evidence_status,
    DROP COLUMN IF EXISTS retrieval_score,
    DROP COLUMN IF EXISTS content_sha256,
    DROP COLUMN IF EXISTS page_start,
    DROP COLUMN IF EXISTS page_end,
    DROP COLUMN IF EXISTS availability;

DROP INDEX IF EXISTS ix_session_source_pins_session_version;
DROP INDEX IF EXISTS ix_session_source_pins_generation;
DROP TABLE IF EXISTS session_source_pins;
DROP TABLE IF EXISTS session_source_pin_sets;
