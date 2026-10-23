-- Reverse only Step 7.5 columns. Keep the 000005 table and its rows.
-- Do not DROP TABLE speech_translation_requests.

DROP INDEX IF EXISTS ix_speech_translation_requests_xa_created;
DROP INDEX IF EXISTS ix_speech_translation_requests_session_created;

ALTER TABLE speech_translation_requests
    DROP CONSTRAINT IF EXISTS speech_translation_requests_outcome_check;

ALTER TABLE speech_translation_requests
    DROP COLUMN IF EXISTS chat_turn_request_id,
    DROP COLUMN IF EXISTS mt_checkpoint_step,
    DROP COLUMN IF EXISTS asr_checkpoint_step,
    DROP COLUMN IF EXISTS generation_matches_rq1,
    DROP COLUMN IF EXISTS checkpoint_identity,
    DROP COLUMN IF EXISTS research_provenance,
    DROP COLUMN IF EXISTS tokenizer_parity,
    DROP COLUMN IF EXISTS model_id,
    DROP COLUMN IF EXISTS duration_ms,
    DROP COLUMN IF EXISTS upstream_status,
    DROP COLUMN IF EXISTS http_status,
    DROP COLUMN IF EXISTS error_code,
    DROP COLUMN IF EXISTS outcome,
    DROP COLUMN IF EXISTS user_id,
    DROP COLUMN IF EXISTS xa_id;
