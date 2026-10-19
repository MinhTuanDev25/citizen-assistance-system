-- 000018_phase5_retrieve.up.sql
-- Session source snapshot + citation evidence snapshot. Additive.

CREATE TABLE IF NOT EXISTS session_source_pin_sets (
    session_id             uuid NOT NULL REFERENCES conversation_sessions (id) ON DELETE CASCADE,
    procedure_version_id   uuid NOT NULL,
    procedure_id           uuid NOT NULL,
    xa_id                  text NOT NULL,
    frozen_at              timestamptz NOT NULL DEFAULT now(),
    materialized           boolean NOT NULL DEFAULT false,
    PRIMARY KEY (session_id, procedure_version_id)
);

CREATE TABLE IF NOT EXISTS session_source_pins (
    session_id             uuid NOT NULL REFERENCES conversation_sessions (id) ON DELETE CASCADE,
    xa_id                  text NOT NULL,
    procedure_id           uuid NOT NULL,
    procedure_version_id   uuid NOT NULL,
    document_id            uuid NOT NULL,
    generation_id          uuid NOT NULL,
    document_title         text NOT NULL DEFAULT '',
    document_number        text,
    content_sha256         text,
    source_sha256          text,
    created_at             timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (session_id, procedure_version_id, document_id),
    CONSTRAINT session_source_pins_generation_fk
        FOREIGN KEY (generation_id)
        REFERENCES document_index_generations (id)
        ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS ix_session_source_pins_generation
    ON session_source_pins (generation_id);

CREATE INDEX IF NOT EXISTS ix_session_source_pins_session_version
    ON session_source_pins (session_id, procedure_version_id);

ALTER TABLE message_citations
    ADD COLUMN IF NOT EXISTS excerpt text,
    ADD COLUMN IF NOT EXISTS excerpt_sha256 text,
    ADD COLUMN IF NOT EXISTS generation_id uuid,
    ADD COLUMN IF NOT EXISTS procedure_version_id uuid,
    ADD COLUMN IF NOT EXISTS evidence_status text,
    ADD COLUMN IF NOT EXISTS retrieval_score double precision,
    ADD COLUMN IF NOT EXISTS content_sha256 text,
    ADD COLUMN IF NOT EXISTS page_start int,
    ADD COLUMN IF NOT EXISTS page_end int,
    ADD COLUMN IF NOT EXISTS availability text;

ALTER TABLE message_citations
    DROP CONSTRAINT IF EXISTS message_citations_evidence_status_check;
ALTER TABLE message_citations
    ADD CONSTRAINT message_citations_evidence_status_check CHECK (
        evidence_status IS NULL
        OR evidence_status IN ('supporting', 'related')
    );

ALTER TABLE message_citations
    DROP CONSTRAINT IF EXISTS message_citations_availability_check;
ALTER TABLE message_citations
    ADD CONSTRAINT message_citations_availability_check CHECK (
        availability IS NULL
        OR availability IN ('available', 'archived', 'unavailable')
    );

ALTER TABLE message_citations
    DROP CONSTRAINT IF EXISTS message_citations_generation_fk;
ALTER TABLE message_citations
    ADD CONSTRAINT message_citations_generation_fk
        FOREIGN KEY (generation_id)
        REFERENCES document_index_generations (id)
        ON DELETE RESTRICT;
