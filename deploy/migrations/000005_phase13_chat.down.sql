-- 000005_phase13_chat.down.sql

DROP INDEX IF EXISTS ix_speech_translation_requests_session;
DROP INDEX IF EXISTS ix_speech_translation_requests_request;
DROP TABLE IF EXISTS speech_translation_requests;
DROP TABLE IF EXISTS model_versions;
DROP INDEX IF EXISTS ix_message_citations_message;
DROP TABLE IF EXISTS message_citations;
DROP INDEX IF EXISTS ux_conversation_messages_session_request_user;
ALTER TABLE conversation_sessions DROP COLUMN IF EXISTS pending_intent;

-- Normalize CONFIRM_INTENT so restoring the pre-phase13 check constraint succeeds with real data.
UPDATE conversation_messages
SET action = 'OUT_OF_SCOPE'
WHERE action = 'CONFIRM_INTENT';

ALTER TABLE conversation_messages DROP CONSTRAINT IF EXISTS conversation_messages_action_check;
ALTER TABLE conversation_messages ADD CONSTRAINT conversation_messages_action_check CHECK (
    action IS NULL
    OR action IN (
        'DIRECT_ANSWER',
        'ASK_MISSING_SLOTS',
        'PROVIDE_FINAL_GUIDANCE',
        'OUT_OF_SCOPE'
    )
);
