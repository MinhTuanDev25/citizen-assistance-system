-- 000004_seed_auth_users.down.sql
-- Keep the seed admin row (needed by procedure_versions.created_by in 000003).
-- Clear any password that may have been set by a local seed-demo command.

UPDATE users
SET password_hash = NULL,
    updated_at = now()
WHERE id = 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee';

DELETE FROM users
WHERE id = 'bbbbbbbb-cccc-dddd-eeee-ffffffffffff';
