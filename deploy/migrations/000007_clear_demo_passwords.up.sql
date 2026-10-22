-- 000007_clear_demo_passwords.up.sql
-- Clear any demo login passwords that older 000004 revisions may have set.
-- Production admin must be bootstrapped via cmd/seed-demo (local) or cmd/bootstrap-admin (env).

UPDATE users
SET password_hash = NULL,
    updated_at = now()
WHERE email IN ('admin@chuse.vn', 'citizen@example.com')
   OR id IN (
        'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee',
        'bbbbbbbb-cccc-dddd-eeee-ffffffffffff'
   );

DELETE FROM users
WHERE id = 'bbbbbbbb-cccc-dddd-eeee-ffffffffffff'
  AND email = 'citizen@example.com';
