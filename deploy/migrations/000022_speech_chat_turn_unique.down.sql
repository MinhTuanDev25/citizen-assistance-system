-- Reverse only the 1-1 chat_turn unique index. Keep 000021 columns.

DROP INDEX IF EXISTS ix_speech_translation_requests_chat_turn_unique;
