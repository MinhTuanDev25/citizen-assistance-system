-- 000017_phase4_activation.up.sql
-- Activation metadata and idempotency action. Does not change seeded ACTIVE versions.

ALTER TABLE procedure_versions
    ADD COLUMN IF NOT EXISTS activated_at timestamptz,
    ADD COLUMN IF NOT EXISTS activated_by uuid REFERENCES users (id) ON DELETE RESTRICT;

ALTER TABLE admin_write_idempotency
    DROP CONSTRAINT IF EXISTS admin_write_idempotency_action_check;

ALTER TABLE admin_write_idempotency
    ADD CONSTRAINT admin_write_idempotency_action_check CHECK (
        action IN ('LINK', 'UNLINK', 'ACTIVATE_VERSION', 'CREATE_VERSION')
    );

CREATE INDEX IF NOT EXISTS ix_audit_logs_activation
    ON audit_logs (entity_type, entity_id, created_at DESC)
    WHERE action = 'PROCEDURE_VERSION_ACTIVATED';
