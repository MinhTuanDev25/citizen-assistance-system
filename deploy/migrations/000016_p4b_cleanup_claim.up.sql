-- 000016_p4b_cleanup_claim.up.sql
-- Cleanup lease so a reaper deletes chunks only after it atomically claims the generation.
-- Existing generations stay PENDING and keep their rows.

ALTER TABLE document_index_generations
    ADD COLUMN cleanup_status text NOT NULL DEFAULT 'PENDING',
    ADD COLUMN cleanup_claim_token uuid,
    ADD COLUMN cleanup_claim_expires_at timestamptz,
    ADD COLUMN cleanup_attempts int NOT NULL DEFAULT 0,
    ADD COLUMN cleanup_completed_at timestamptz,
    ADD COLUMN cleanup_error text;

ALTER TABLE document_index_generations
    ADD CONSTRAINT document_index_generations_cleanup_status_check CHECK (
        cleanup_status IN ('PENDING', 'CLAIMED', 'COMPLETED', 'RETRYABLE_FAILED', 'TERMINAL_FAILED')
    ),
    ADD CONSTRAINT document_index_generations_cleanup_attempts_check CHECK (cleanup_attempts >= 0),
    ADD CONSTRAINT document_index_generations_cleanup_error_check CHECK (
        cleanup_error IS NULL OR char_length(cleanup_error) BETWEEN 1 AND 64
    ),
    ADD CONSTRAINT document_index_generations_cleanup_shape_check CHECK (
        (
            cleanup_status = 'PENDING'
            AND cleanup_claim_token IS NULL
            AND cleanup_claim_expires_at IS NULL
            AND cleanup_completed_at IS NULL
        )
        OR (
            cleanup_status = 'CLAIMED'
            AND cleanup_claim_token IS NOT NULL
            AND cleanup_claim_expires_at IS NOT NULL
            AND cleanup_completed_at IS NULL
        )
        OR (
            cleanup_status = 'COMPLETED'
            AND cleanup_claim_token IS NULL
            AND cleanup_claim_expires_at IS NULL
            AND cleanup_completed_at IS NOT NULL
        )
        OR (
            cleanup_status IN ('RETRYABLE_FAILED', 'TERMINAL_FAILED')
            AND cleanup_claim_token IS NULL
            AND cleanup_claim_expires_at IS NULL
            AND cleanup_completed_at IS NULL
        )
    );
