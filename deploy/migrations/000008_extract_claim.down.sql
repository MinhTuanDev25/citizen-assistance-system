-- 000008_extract_claim.down.sql
-- Drop in-flight claims first so response_json can become NOT NULL again.
-- Completed envelopes are kept.

DELETE FROM turn_idempotency WHERE status = 'PENDING';

ALTER TABLE turn_idempotency DROP CONSTRAINT IF EXISTS turn_idempotency_payload_chk;
ALTER TABLE turn_idempotency DROP CONSTRAINT IF EXISTS turn_idempotency_status_chk;

ALTER TABLE turn_idempotency
    ALTER COLUMN response_json SET NOT NULL;

ALTER TABLE turn_idempotency DROP COLUMN IF EXISTS claim_expires_at;
ALTER TABLE turn_idempotency DROP COLUMN IF EXISTS claimed_at;
ALTER TABLE turn_idempotency DROP COLUMN IF EXISTS status;
