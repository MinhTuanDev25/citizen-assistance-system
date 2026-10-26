-- 000022_speech_chat_turn_unique.up.sql
-- One Chat Turn X-Request-ID may appear on at most one speech audit row.
-- NULL chat_turn_request_id values are unchanged (historical unlink stays).
-- Fails loudly if duplicates already exist; do not backfill.

CREATE UNIQUE INDEX ix_speech_translation_requests_chat_turn_unique
    ON speech_translation_requests (chat_turn_request_id)
    WHERE chat_turn_request_id IS NOT NULL;
