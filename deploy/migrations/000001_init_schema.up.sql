-- 000001_init_schema.up.sql
-- Citizen Assistance System V1 — matches docs/phase-0/db/data-model.md (frozen)

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE communes (
    id          text PRIMARY KEY,
    name        text NOT NULL,
    description text,
    is_active   boolean NOT NULL DEFAULT true,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE domains (
    id          text PRIMARY KEY,
    name        text NOT NULL,
    description text,
    sort_order  int NOT NULL DEFAULT 0,
    is_active   boolean NOT NULL DEFAULT true,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE users (
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    role          text NOT NULL,
    full_name     text NOT NULL,
    email         text,
    phone         text,
    password_hash text,
    created_at    timestamptz NOT NULL DEFAULT now(),
    updated_at    timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT users_role_check CHECK (role IN ('CITIZEN', 'ADMIN'))
);

CREATE UNIQUE INDEX ux_users_email ON users (lower(email)) WHERE email IS NOT NULL;

CREATE TABLE procedures (
    id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    procedure_code    text NOT NULL,
    domain_id         text NOT NULL REFERENCES domains (id) ON DELETE RESTRICT,
    name              text NOT NULL,
    xa_id             text NOT NULL REFERENCES communes (id) ON DELETE RESTRICT,
    active_version_id uuid,
    created_at        timestamptz NOT NULL DEFAULT now(),
    updated_at        timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT ux_procedures_xa_code UNIQUE (xa_id, procedure_code),
    CONSTRAINT ux_procedures_id_xa UNIQUE (id, xa_id)
);

CREATE TABLE documents (
    id                 uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    xa_id              text NOT NULL REFERENCES communes (id) ON DELETE RESTRICT,
    domain_id          text REFERENCES domains (id) ON DELETE RESTRICT,
    title              text NOT NULL,
    document_number    text,
    issuer             text,
    filename           text NOT NULL,
    storage_uri        text NOT NULL,
    checksum           text NOT NULL,
    effective_date     date,
    expire_date        date,
    issued_date        date,
    processing_status  text NOT NULL DEFAULT 'UPLOADED',
    validity_status    text NOT NULL DEFAULT 'PENDING',
    uploaded_by        uuid NOT NULL REFERENCES users (id) ON DELETE RESTRICT,
    created_at         timestamptz NOT NULL DEFAULT now(),
    updated_at         timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT documents_processing_status_check CHECK (
        processing_status IN ('UPLOADED', 'PROCESSING', 'PROCESSED', 'FAILED')
    ),
    CONSTRAINT documents_validity_status_check CHECK (
        validity_status IN ('PENDING', 'VALID', 'EXPIRED', 'SUPERSEDED')
    ),
    CONSTRAINT documents_dates_check CHECK (
        effective_date IS NULL
        OR expire_date IS NULL
        OR expire_date >= effective_date
    )
);

CREATE INDEX ix_documents_xa_status ON documents (xa_id, processing_status, validity_status);
CREATE INDEX ix_documents_checksum ON documents (checksum);

CREATE TABLE procedure_drafts (
    id                 uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id        uuid REFERENCES documents (id) ON DELETE RESTRICT,
    procedure_id       uuid REFERENCES procedures (id) ON DELETE RESTRICT,
    draft_definition   jsonb NOT NULL DEFAULT '{}'::jsonb,
    validation_result  jsonb,
    status             text NOT NULL DEFAULT 'DRAFT',
    created_by         uuid NOT NULL REFERENCES users (id) ON DELETE RESTRICT,
    updated_by         uuid NOT NULL REFERENCES users (id) ON DELETE RESTRICT,
    created_at         timestamptz NOT NULL DEFAULT now(),
    updated_at         timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT procedure_drafts_status_check CHECK (
        status IN ('DRAFT', 'REVIEWED', 'APPROVED', 'REJECTED', 'PUBLISHED')
    )
);

CREATE TABLE procedure_versions (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    procedure_id    uuid NOT NULL REFERENCES procedures (id) ON DELETE RESTRICT,
    version         text NOT NULL,
    status          text NOT NULL,
    definition      jsonb NOT NULL DEFAULT '{}'::jsonb,
    source_draft_id uuid REFERENCES procedure_drafts (id) ON DELETE RESTRICT,
    created_by      uuid NOT NULL REFERENCES users (id) ON DELETE RESTRICT,
    approved_by     uuid REFERENCES users (id) ON DELETE RESTRICT,
    created_at      timestamptz NOT NULL DEFAULT now(),
    approved_at     timestamptz,
    CONSTRAINT procedure_versions_status_check CHECK (
        status IN ('APPROVED', 'INDEXING', 'ACTIVE', 'ARCHIVED')
    ),
    CONSTRAINT ux_procedure_versions_proc_semver UNIQUE (procedure_id, version),
    CONSTRAINT ux_procedure_versions_proc_id UNIQUE (procedure_id, id)
);

CREATE UNIQUE INDEX ux_procedure_versions_one_active
    ON procedure_versions (procedure_id)
    WHERE status = 'ACTIVE';

CREATE UNIQUE INDEX ux_procedure_versions_source_draft
    ON procedure_versions (source_draft_id)
    WHERE source_draft_id IS NOT NULL;

ALTER TABLE procedures
    ADD CONSTRAINT fk_procedures_active_version
    FOREIGN KEY (id, active_version_id)
    REFERENCES procedure_versions (procedure_id, id)
    ON DELETE RESTRICT;

CREATE TABLE procedure_version_documents (
    procedure_version_id uuid NOT NULL REFERENCES procedure_versions (id) ON DELETE RESTRICT,
    document_id          uuid NOT NULL REFERENCES documents (id) ON DELETE RESTRICT,
    PRIMARY KEY (procedure_version_id, document_id)
);

CREATE TABLE conversation_sessions (
    id                           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id                      uuid REFERENCES users (id) ON DELETE SET NULL,
    guest_token                  text,
    xa_id                        text NOT NULL REFERENCES communes (id) ON DELETE RESTRICT,
    active_procedure_id          uuid,
    active_procedure_version_id  uuid,
    status                       text NOT NULL DEFAULT 'OPEN',
    created_at                   timestamptz NOT NULL DEFAULT now(),
    updated_at                   timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT conversation_sessions_status_check CHECK (
        status IN ('OPEN', 'COMPLETED', 'ABANDONED')
    ),
    CONSTRAINT conversation_sessions_active_pair_check CHECK (
        (active_procedure_id IS NULL AND active_procedure_version_id IS NULL)
        OR (active_procedure_id IS NOT NULL AND active_procedure_version_id IS NOT NULL)
    ),
    CONSTRAINT fk_sessions_active_version
        FOREIGN KEY (active_procedure_id, active_procedure_version_id)
        REFERENCES procedure_versions (procedure_id, id)
        ON DELETE RESTRICT,
    CONSTRAINT fk_sessions_active_procedure_xa
        FOREIGN KEY (active_procedure_id, xa_id)
        REFERENCES procedures (id, xa_id)
        ON DELETE RESTRICT
);

CREATE UNIQUE INDEX ux_conversation_sessions_guest_token
    ON conversation_sessions (guest_token)
    WHERE guest_token IS NOT NULL;

CREATE INDEX ix_conversation_sessions_user_updated
    ON conversation_sessions (user_id, updated_at DESC);

CREATE TABLE conversation_messages (
    id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id       uuid NOT NULL REFERENCES conversation_sessions (id) ON DELETE CASCADE,
    request_id       uuid NOT NULL,
    role             text NOT NULL,
    content          text NOT NULL,
    action           text,
    message_metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at       timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT conversation_messages_role_check CHECK (
        role IN ('USER', 'ASSISTANT', 'SYSTEM')
    ),
    CONSTRAINT conversation_messages_action_check CHECK (
        action IS NULL
        OR action IN (
            'DIRECT_ANSWER',
            'ASK_MISSING_SLOTS',
            'PROVIDE_FINAL_GUIDANCE',
            'OUT_OF_SCOPE'
        )
    )
);

CREATE INDEX ix_conversation_messages_request_id ON conversation_messages (request_id);
CREATE INDEX ix_conversation_messages_session_created
    ON conversation_messages (session_id, created_at);

CREATE TABLE session_slot_states (
    session_id           uuid NOT NULL REFERENCES conversation_sessions (id) ON DELETE CASCADE,
    procedure_id         uuid NOT NULL,
    procedure_version_id uuid NOT NULL,
    slot_state           jsonb NOT NULL DEFAULT '{}'::jsonb,
    updated_at           timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (session_id, procedure_id),
    CONSTRAINT fk_slot_states_version
        FOREIGN KEY (procedure_id, procedure_version_id)
        REFERENCES procedure_versions (procedure_id, id)
        ON DELETE RESTRICT
);

CREATE TABLE knowledge_chunks (
    id                   uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    xa_id                text NOT NULL REFERENCES communes (id) ON DELETE RESTRICT,
    procedure_id         uuid NOT NULL,
    procedure_version_id uuid NOT NULL,
    document_id          uuid NOT NULL REFERENCES documents (id) ON DELETE RESTRICT,
    chunk_index          int NOT NULL,
    content              text NOT NULL,
    metadata             jsonb NOT NULL DEFAULT '{}'::jsonb,
    embedding            vector(1536) NOT NULL,
    created_at           timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT knowledge_chunks_chunk_index_check CHECK (chunk_index >= 0),
    CONSTRAINT ux_knowledge_chunks_version_doc_idx UNIQUE (procedure_version_id, document_id, chunk_index),
    CONSTRAINT fk_knowledge_chunks_version
        FOREIGN KEY (procedure_id, procedure_version_id)
        REFERENCES procedure_versions (procedure_id, id)
        ON DELETE RESTRICT,
    CONSTRAINT fk_knowledge_chunks_procedure_xa
        FOREIGN KEY (procedure_id, xa_id)
        REFERENCES procedures (id, xa_id)
        ON DELETE RESTRICT
);

CREATE INDEX ix_knowledge_chunks_filter
    ON knowledge_chunks (xa_id, procedure_id, procedure_version_id);

CREATE INDEX ix_knowledge_chunks_embedding_hnsw
    ON knowledge_chunks
    USING hnsw (embedding vector_cosine_ops);

CREATE TABLE audit_logs (
    id            bigserial PRIMARY KEY,
    request_id    uuid,
    actor_user_id uuid REFERENCES users (id) ON DELETE SET NULL,
    action        text NOT NULL,
    entity_type   text,
    entity_id     text,
    payload       jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX ix_audit_logs_request_id ON audit_logs (request_id);
CREATE INDEX ix_audit_logs_created_at ON audit_logs (created_at DESC);
CREATE INDEX ix_audit_logs_entity ON audit_logs (entity_type, entity_id);
