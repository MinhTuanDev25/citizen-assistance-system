-- Restore the 000022 status and action sets.
-- Fails loudly while SOURCE versions or CREATE_PROCEDURE idempotency rows exist;
-- this migration does not delete procedures an admin created.

ALTER TABLE admin_write_idempotency DROP CONSTRAINT admin_write_idempotency_action_check;
ALTER TABLE admin_write_idempotency ADD CONSTRAINT admin_write_idempotency_action_check CHECK (
    action IN ('LINK', 'UNLINK', 'ACTIVATE_VERSION', 'CREATE_VERSION')
);

ALTER TABLE procedure_versions DROP CONSTRAINT procedure_versions_status_check;
ALTER TABLE procedure_versions ADD CONSTRAINT procedure_versions_status_check CHECK (
    status IN ('APPROVED', 'INDEXING', 'ACTIVE', 'ARCHIVED')
);
