-- 000023_procedure_source_version.up.sql
-- Admin can create a procedure from an empty database.
-- A new procedure gets one SOURCE version: an anchor for linking and indexing
-- the first PDF so an AI draft can be generated. SOURCE has an empty definition
-- and is never activatable (eligibility and Activate only accept APPROVED/INDEXING).
-- CREATE_PROCEDURE is the idempotency action for POST /admin/procedures.

ALTER TABLE procedure_versions DROP CONSTRAINT procedure_versions_status_check;
ALTER TABLE procedure_versions ADD CONSTRAINT procedure_versions_status_check CHECK (
    status IN ('SOURCE', 'APPROVED', 'INDEXING', 'ACTIVE', 'ARCHIVED')
);

ALTER TABLE admin_write_idempotency DROP CONSTRAINT admin_write_idempotency_action_check;
ALTER TABLE admin_write_idempotency ADD CONSTRAINT admin_write_idempotency_action_check CHECK (
    action IN ('LINK', 'UNLINK', 'ACTIVATE_VERSION', 'CREATE_VERSION', 'CREATE_PROCEDURE')
);
