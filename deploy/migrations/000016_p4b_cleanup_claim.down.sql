ALTER TABLE document_index_generations
    DROP CONSTRAINT IF EXISTS document_index_generations_cleanup_shape_check,
    DROP CONSTRAINT IF EXISTS document_index_generations_cleanup_error_check,
    DROP CONSTRAINT IF EXISTS document_index_generations_cleanup_attempts_check,
    DROP CONSTRAINT IF EXISTS document_index_generations_cleanup_status_check;

ALTER TABLE document_index_generations
    DROP COLUMN IF EXISTS cleanup_error,
    DROP COLUMN IF EXISTS cleanup_completed_at,
    DROP COLUMN IF EXISTS cleanup_attempts,
    DROP COLUMN IF EXISTS cleanup_claim_expires_at,
    DROP COLUMN IF EXISTS cleanup_claim_token,
    DROP COLUMN IF EXISTS cleanup_status;
