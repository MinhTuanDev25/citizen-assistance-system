-- 000010_document_indexing.up.sql
-- P4A indexing foundation. Does not embed, OCR, or activate.
-- READY is added beside PROCESSED. Supersede stays inside one commune.
-- A link row can exist only when the document, procedure, and version
-- share xa_id and domain_id.

ALTER TABLE documents DROP CONSTRAINT documents_processing_status_check;

ALTER TABLE documents
    ADD CONSTRAINT documents_processing_status_check CHECK (
        processing_status IN ('UPLOADED', 'PROCESSING', 'PROCESSED', 'READY', 'FAILED')
    );

ALTER TABLE documents
    ADD COLUMN supersedes_document_id uuid;

ALTER TABLE documents
    ADD CONSTRAINT ux_documents_id_xa UNIQUE (id, xa_id);

ALTER TABLE documents
    ADD CONSTRAINT ux_documents_id_xa_domain UNIQUE (id, xa_id, domain_id);

ALTER TABLE documents
    ADD CONSTRAINT documents_not_self_supersede
        CHECK (supersedes_document_id IS DISTINCT FROM id);

ALTER TABLE documents
    ADD CONSTRAINT fk_documents_supersedes
        FOREIGN KEY (supersedes_document_id, xa_id)
        REFERENCES documents (id, xa_id)
        ON DELETE RESTRICT;

ALTER TABLE procedures
    ADD CONSTRAINT ux_procedures_id_xa_domain UNIQUE (id, xa_id, domain_id);

ALTER TABLE procedure_version_documents
    ADD COLUMN procedure_id uuid,
    ADD COLUMN xa_id text,
    ADD COLUMN domain_id text,
    ADD COLUMN relationship_type text,
    ADD COLUMN page_range text,
    ADD COLUMN created_at timestamptz NOT NULL DEFAULT now();

UPDATE procedure_version_documents AS link
SET procedure_id = version.procedure_id,
    xa_id = proc.xa_id,
    domain_id = proc.domain_id,
    relationship_type = 'SOURCE'
FROM procedure_versions AS version
JOIN procedures AS proc ON proc.id = version.procedure_id
WHERE version.id = link.procedure_version_id;

ALTER TABLE procedure_version_documents
    ALTER COLUMN procedure_id SET NOT NULL,
    ALTER COLUMN xa_id SET NOT NULL,
    ALTER COLUMN domain_id SET NOT NULL,
    ALTER COLUMN relationship_type SET NOT NULL;

ALTER TABLE procedure_version_documents
    ADD CONSTRAINT pvd_relationship_type_check CHECK (
        relationship_type IN ('SOURCE', 'SUPERSEDES')
    ),
    ADD CONSTRAINT pvd_page_range_check CHECK (
        page_range IS NULL
        OR page_range ~ '^[1-9][0-9]{0,3}(-[1-9][0-9]{0,3})?$'
    );

ALTER TABLE procedure_version_documents
    ADD CONSTRAINT fk_pvd_version
        FOREIGN KEY (procedure_id, procedure_version_id)
        REFERENCES procedure_versions (procedure_id, id)
        ON DELETE RESTRICT,
    ADD CONSTRAINT fk_pvd_procedure_xa_domain
        FOREIGN KEY (procedure_id, xa_id, domain_id)
        REFERENCES procedures (id, xa_id, domain_id)
        ON DELETE RESTRICT,
    ADD CONSTRAINT fk_pvd_document_xa_domain
        FOREIGN KEY (document_id, xa_id, domain_id)
        REFERENCES documents (id, xa_id, domain_id)
        ON DELETE RESTRICT;

CREATE INDEX ix_pvd_document ON procedure_version_documents (document_id);

CREATE TABLE document_index_jobs (
    id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id       uuid NOT NULL,
    xa_id             text NOT NULL,
    request_id        uuid NOT NULL,
    payload_hash      text NOT NULL,
    status            text NOT NULL,
    error_code        text,
    claimed_at        timestamptz,
    claim_expires_at  timestamptz,
    finished_at       timestamptz,
    created_at        timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT document_index_jobs_status_check CHECK (
        status IN ('CLAIMED', 'SUCCEEDED', 'FAILED')
    ),
    CONSTRAINT document_index_jobs_claim_check CHECK (
        (
            status = 'CLAIMED'
            AND claimed_at IS NOT NULL
            AND claim_expires_at IS NOT NULL
            AND finished_at IS NULL
        )
        OR (
            status IN ('SUCCEEDED', 'FAILED')
            AND finished_at IS NOT NULL
            AND claim_expires_at IS NULL
        )
    ),
    CONSTRAINT fk_index_jobs_document_xa
        FOREIGN KEY (document_id, xa_id)
        REFERENCES documents (id, xa_id)
        ON DELETE RESTRICT,
    CONSTRAINT ux_document_index_jobs_request UNIQUE (document_id, request_id)
);

CREATE UNIQUE INDEX ux_document_index_jobs_one_claim
    ON document_index_jobs (document_id)
    WHERE status = 'CLAIMED';

CREATE TABLE admin_write_idempotency (
    request_id    uuid NOT NULL,
    action        text NOT NULL,
    payload_hash  text NOT NULL,
    status_code   int NOT NULL,
    response_json jsonb NOT NULL,
    created_at    timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (request_id, action),
    CONSTRAINT admin_write_idempotency_action_check CHECK (
        action IN ('LINK', 'UNLINK')
    )
);
