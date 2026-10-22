-- 000017_phase4_activation.down.sql
-- Drops activation columns only. Idempotency/audit rows are kept.
-- Recreating the old action check fails if ACTIVATE_VERSION or CREATE_VERSION
-- rows exist; that is a forward-fix, not a silent history wipe.

ALTER TABLE procedure_versions
    DROP COLUMN IF EXISTS activated_at,
    DROP COLUMN IF EXISTS activated_by;

DROP INDEX IF EXISTS ix_audit_logs_activation;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM admin_write_idempotency
        WHERE action IN ('ACTIVATE_VERSION', 'CREATE_VERSION')
    ) THEN
        RAISE EXCEPTION '000017 down is blocked while ACTIVATE_VERSION/CREATE_VERSION idempotency rows exist; keep the expanded action check (forward-fix)';
    END IF;
END $$;

ALTER TABLE admin_write_idempotency
    DROP CONSTRAINT IF EXISTS admin_write_idempotency_action_check;

ALTER TABLE admin_write_idempotency
    ADD CONSTRAINT admin_write_idempotency_action_check CHECK (
        action IN ('LINK', 'UNLINK')
    );
