-- 000006_idempotency_envelope.down.sql
DROP INDEX IF EXISTS ix_turn_idempotency_created;
DROP TABLE IF EXISTS turn_idempotency;
