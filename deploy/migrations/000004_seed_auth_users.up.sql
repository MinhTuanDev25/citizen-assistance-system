-- 000004_seed_auth_users.up.sql
-- Production-safe: ensure seed admin exists for FK (created_by) WITHOUT a login password.
-- Demo passwords are NOT set here. Use: go run ./cmd/seed-demo (APP_ENV=local only).

INSERT INTO users (id, role, full_name, email, password_hash)
VALUES (
    'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee',
    'ADMIN',
    'Seed Admin',
    'admin@chuse.vn',
    NULL
)
ON CONFLICT (id) DO UPDATE
SET
    role = 'ADMIN',
    full_name = COALESCE(users.full_name, EXCLUDED.full_name),
    email = EXCLUDED.email,
    -- Never invent a default password in migrations.
    password_hash = NULL,
    updated_at = now();
