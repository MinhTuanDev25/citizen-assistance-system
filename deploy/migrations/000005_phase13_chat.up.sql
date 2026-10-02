-- 000005_phase13_chat.up.sql
-- Phase 1–3: citations, speech trace tables, turn idempotency, pending intent

-- Allow CONFIRM_INTENT on assistant messages
ALTER TABLE conversation_messages DROP CONSTRAINT IF EXISTS conversation_messages_action_check;
ALTER TABLE conversation_messages ADD CONSTRAINT conversation_messages_action_check CHECK (
    action IS NULL
    OR action IN (
        'DIRECT_ANSWER',
        'ASK_MISSING_SLOTS',
        'PROVIDE_FINAL_GUIDANCE',
        'OUT_OF_SCOPE',
        'CONFIRM_INTENT'
    )
);

ALTER TABLE conversation_sessions
    ADD COLUMN IF NOT EXISTS pending_intent jsonb NOT NULL DEFAULT '{}'::jsonb;

-- One completed turn = one USER row per (session, request_id). ASSISTANT shares request_id.
CREATE UNIQUE INDEX IF NOT EXISTS ux_conversation_messages_session_request_user
    ON conversation_messages (session_id, request_id)
    WHERE role = 'USER';

CREATE TABLE IF NOT EXISTS message_citations (
    id                 uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    message_id         uuid NOT NULL REFERENCES conversation_messages (id) ON DELETE CASCADE,
    knowledge_chunk_id uuid REFERENCES knowledge_chunks (id) ON DELETE SET NULL,
    document_id        uuid REFERENCES documents (id) ON DELETE SET NULL,
    title              text,
    source_type        text,
    page_range         text,
    doc_id             text,
    effective_date     text,
    issuer             text,
    created_at         timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_message_citations_message
    ON message_citations (message_id);

CREATE TABLE IF NOT EXISTS model_versions (
    id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    architecture   text NOT NULL,
    checkpoint_uri text NOT NULL,
    config         jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at     timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT model_versions_architecture_check CHECK (
        architecture IN ('cascaded', 'direct', 'llm_extract', 'embedding', 'other')
    )
);

CREATE TABLE IF NOT EXISTS speech_translation_requests (
    id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id       uuid NOT NULL,
    session_id       uuid REFERENCES conversation_sessions (id) ON DELETE SET NULL,
    model_version_id uuid REFERENCES model_versions (id) ON DELETE RESTRICT,
    audio_uri        text,
    transcript_bahnar text,
    output_vi        text,
    latency_ms       int,
    status           text NOT NULL DEFAULT 'QUEUED',
    created_at       timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT speech_translation_requests_status_check CHECK (
        status IN ('QUEUED', 'OK', 'FAILED')
    )
);

CREATE INDEX IF NOT EXISTS ix_speech_translation_requests_request
    ON speech_translation_requests (request_id);

CREATE INDEX IF NOT EXISTS ix_speech_translation_requests_session
    ON speech_translation_requests (session_id);
