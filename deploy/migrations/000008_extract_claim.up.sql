-- 000008_extract_claim.up.sql
-- Atomic PENDING claim so two concurrent turns with the same
-- (session_id, request_id) call the extractor at most once.
-- The claim row is written in a short transaction and released before the
-- AI network call. claim_expires_at is the lease: an abandoned PENDING row
-- can be taken over once it is in the past.
-- Existing rows are completed turns and stay COMPLETE with a body.

ALTER TABLE turn_idempotency
    ADD COLUMN status text NOT NULL DEFAULT 'COMPLETE',
    ADD COLUMN claimed_at timestamptz,
    ADD COLUMN claim_expires_at timestamptz;

ALTER TABLE turn_idempotency
    ALTER COLUMN response_json DROP NOT NULL;

ALTER TABLE turn_idempotency
    ADD CONSTRAINT turn_idempotency_status_chk
        CHECK (status IN ('PENDING', 'COMPLETE'));

ALTER TABLE turn_idempotency
    ADD CONSTRAINT turn_idempotency_payload_chk
        CHECK (
            (status = 'COMPLETE' AND response_json IS NOT NULL AND claim_expires_at IS NULL)
            OR
            (status = 'PENDING' AND response_json IS NULL AND claimed_at IS NOT NULL AND claim_expires_at IS NOT NULL)
        );
