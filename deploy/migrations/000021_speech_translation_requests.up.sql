-- 000021_speech_translation_requests.up.sql
-- Step 7.5: evolve the Phase 1.3 placeholder created in 000005.
-- Case A: same audit domain. Additive only. Does not CREATE or DROP the table.
-- Does not backfill or delete audio_uri / transcript_bahnar / output_vi.
-- New Go writes never populate those columns.
-- Version 000021 was never applied (local schema_migrations.version = 20).
-- Columns are added without IF NOT EXISTS so an incompatible existing
-- column fails loudly instead of being skipped.

ALTER TABLE speech_translation_requests
    ADD COLUMN xa_id text REFERENCES communes (id) ON DELETE RESTRICT,
    ADD COLUMN user_id uuid REFERENCES users (id) ON DELETE SET NULL,
    ADD COLUMN outcome text,
    ADD COLUMN error_code text,
    ADD COLUMN http_status int,
    ADD COLUMN upstream_status int,
    ADD COLUMN duration_ms int,
    ADD COLUMN model_id text,
    ADD COLUMN tokenizer_parity text,
    ADD COLUMN research_provenance text,
    ADD COLUMN checkpoint_identity text,
    ADD COLUMN generation_matches_rq1 boolean,
    ADD COLUMN asr_checkpoint_step int,
    ADD COLUMN mt_checkpoint_step int,
    ADD COLUMN chat_turn_request_id uuid;

UPDATE speech_translation_requests AS r
SET xa_id = s.xa_id
FROM conversation_sessions AS s
WHERE r.session_id = s.id
  AND r.xa_id IS NULL
  AND s.xa_id IS NOT NULL;

UPDATE speech_translation_requests
SET outcome = CASE status
    WHEN 'OK' THEN 'SUCCESS'
    WHEN 'FAILED' THEN 'FAILURE'
    ELSE outcome
END
WHERE outcome IS NULL;

ALTER TABLE speech_translation_requests
    DROP CONSTRAINT IF EXISTS speech_translation_requests_outcome_check;

ALTER TABLE speech_translation_requests
    ADD CONSTRAINT speech_translation_requests_outcome_check
    CHECK (outcome IS NULL OR outcome IN ('SUCCESS', 'FAILURE'));

CREATE INDEX IF NOT EXISTS ix_speech_translation_requests_session_created
    ON speech_translation_requests (session_id, created_at DESC);

CREATE INDEX IF NOT EXISTS ix_speech_translation_requests_xa_created
    ON speech_translation_requests (xa_id, created_at DESC);
