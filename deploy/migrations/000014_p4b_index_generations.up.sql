-- 000014_p4b_index_generations.up.sql
-- Staging generations and chunk rows for one document–version link.
-- Vectors stay in Qdrant. This file does not activate a procedure version.

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM knowledge_chunks) THEN
        RAISE EXCEPTION 'knowledge_chunks is not empty; 000014 will not guess a generation';
    END IF;
END $$;

CREATE TABLE document_index_generations (
    id                     uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    xa_id                  text NOT NULL,
    document_id            uuid NOT NULL,
    procedure_version_id   uuid NOT NULL,
    job_id                 uuid NOT NULL REFERENCES document_index_jobs (id) ON DELETE RESTRICT,
    status                 text NOT NULL,
    pipeline_version       text NOT NULL,
    extraction_version     text NOT NULL,
    ocr_version            text NOT NULL,
    chunk_config_hash      text NOT NULL,
    embedding_model_id     text NOT NULL,
    embedding_revision     text NOT NULL,
    embedding_checksum     text NOT NULL,
    vector_dimension       int NOT NULL,
    source_sha256          text NOT NULL,
    content_sha256         text NOT NULL,
    manifest_hash          text NOT NULL,
    page_count             int NOT NULL DEFAULT 0,
    native_page_count      int NOT NULL DEFAULT 0,
    ocr_page_count         int NOT NULL DEFAULT 0,
    chunk_count            int NOT NULL DEFAULT 0,
    vector_count           int NOT NULL DEFAULT 0,
    error_code             text,
    created_at             timestamptz NOT NULL DEFAULT now(),
    updated_at             timestamptz NOT NULL DEFAULT now(),
    published_at           timestamptz,
    CONSTRAINT document_index_generations_status_check CHECK (
        status IN ('STAGING', 'READY', 'SUPERSEDED', 'FAILED')
    ),
    CONSTRAINT document_index_generations_counts_check CHECK (
        page_count >= 0 AND native_page_count >= 0 AND ocr_page_count >= 0
        AND chunk_count >= 0 AND vector_count >= 0 AND vector_dimension >= 0
        AND native_page_count + ocr_page_count <= page_count
    ),
    CONSTRAINT document_index_generations_sha_check CHECK (
        source_sha256 ~ '^[0-9a-f]{64}$'
        AND (content_sha256 = '' OR content_sha256 ~ '^[0-9a-f]{64}$')
        AND (manifest_hash = '' OR manifest_hash ~ '^[0-9a-f]{64}$')
        AND (embedding_checksum = '' OR embedding_checksum ~ '^[0-9a-f]{64}$')
    ),
    CONSTRAINT document_index_generations_error_check CHECK (
        error_code IS NULL OR char_length(error_code) BETWEEN 1 AND 64
    ),
    CONSTRAINT fk_generations_document_xa
        FOREIGN KEY (document_id, xa_id) REFERENCES documents (id, xa_id) ON DELETE RESTRICT,
    CONSTRAINT fk_generations_link
        FOREIGN KEY (procedure_version_id, document_id)
        REFERENCES procedure_version_documents (procedure_version_id, document_id) ON DELETE RESTRICT
);

CREATE UNIQUE INDEX ux_generation_one_ready
    ON document_index_generations (document_id, procedure_version_id)
    WHERE status = 'READY';

ALTER TABLE document_index_generations
    ADD CONSTRAINT ux_generations_id_link
        UNIQUE (id, document_id, procedure_version_id, xa_id);

ALTER TABLE procedure_version_documents
    ADD COLUMN active_generation_id uuid;

ALTER TABLE procedure_version_documents
    ADD CONSTRAINT fk_pvd_active_generation
        FOREIGN KEY (active_generation_id, document_id, procedure_version_id, xa_id)
        REFERENCES document_index_generations (id, document_id, procedure_version_id, xa_id)
        ON DELETE RESTRICT;

ALTER TABLE knowledge_chunks
    ALTER COLUMN embedding DROP NOT NULL;

ALTER TABLE knowledge_chunks
    ADD COLUMN generation_id uuid,
    ADD COLUMN page_start int,
    ADD COLUMN page_end int,
    ADD COLUMN text_sha256 text,
    ADD COLUMN token_count int,
    ADD COLUMN extraction_source text;

ALTER TABLE knowledge_chunks
    ALTER COLUMN generation_id SET NOT NULL,
    ALTER COLUMN page_start SET NOT NULL,
    ALTER COLUMN page_end SET NOT NULL,
    ALTER COLUMN text_sha256 SET NOT NULL,
    ALTER COLUMN token_count SET NOT NULL,
    ALTER COLUMN extraction_source SET NOT NULL;

ALTER TABLE knowledge_chunks
    DROP CONSTRAINT ux_knowledge_chunks_version_doc_idx;

ALTER TABLE knowledge_chunks
    ADD CONSTRAINT knowledge_chunks_page_check CHECK (page_start >= 1 AND page_end >= page_start),
    ADD CONSTRAINT knowledge_chunks_token_check CHECK (token_count >= 0),
    ADD CONSTRAINT knowledge_chunks_source_check CHECK (extraction_source IN ('native', 'ocr', 'mixed')),
    ADD CONSTRAINT knowledge_chunks_text_sha_check CHECK (text_sha256 ~ '^[0-9a-f]{64}$'),
    ADD CONSTRAINT ux_knowledge_chunks_generation_idx UNIQUE (generation_id, chunk_index);

ALTER TABLE document_index_generations
    ADD CONSTRAINT ux_generations_id_xa UNIQUE (id, xa_id);

ALTER TABLE knowledge_chunks
    ADD CONSTRAINT fk_knowledge_chunks_generation_xa
        FOREIGN KEY (generation_id, xa_id)
        REFERENCES document_index_generations (id, xa_id)
        ON DELETE RESTRICT;
