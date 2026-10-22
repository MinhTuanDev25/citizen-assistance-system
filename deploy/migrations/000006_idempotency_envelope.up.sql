-- 000006_idempotency_envelope.up.sql
-- Persist payload hash + full turn response for idempotent replay

CREATE TABLE IF NOT EXISTS turn_idempotency (
    session_id     uuid NOT NULL REFERENCES conversation_sessions (id) ON DELETE CASCADE,
    request_id     uuid NOT NULL,
    payload_hash   text NOT NULL,
    response_json  jsonb NOT NULL,
    created_at     timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (session_id, request_id)
);

CREATE INDEX IF NOT EXISTS ix_turn_idempotency_created
    ON turn_idempotency (created_at);
