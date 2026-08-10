-- 000004_seed_auth_users.down.sql

DELETE FROM users WHERE id = 'bbbbbbbb-cccc-dddd-eeee-ffffffffffff';

UPDATE users
SET
    password_hash = NULL,
    full_name = 'Seed Admin',
    updated_at = now()
WHERE id = 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee';
