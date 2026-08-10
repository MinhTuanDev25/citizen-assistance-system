-- 000004_seed_auth_users.up.sql
-- Password hashes (bcrypt cost 10): admin123 / citizen123 — local demo only.

UPDATE users
SET
    password_hash = '$2a$10$Ns59hUCLGDWnXSJ9Up1sauMCJUw/WqWdBll1FGUFuszjajDo4dblC',
    full_name = 'Cán bộ One Cửa',
    updated_at = now()
WHERE id = 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'
  AND email = 'admin@chuse.vn';

INSERT INTO users (id, role, full_name, email, password_hash)
VALUES (
    'bbbbbbbb-cccc-dddd-eeee-ffffffffffff',
    'CITIZEN',
    'Công dân demo',
    'citizen@example.com',
    '$2a$10$vB2NiXGJ.8UmDMgu22fA/uyMd0X53wrrUFoiNsx8sMQ8orTvnRxdO'
)
ON CONFLICT (id) DO UPDATE
SET
    role = EXCLUDED.role,
    full_name = EXCLUDED.full_name,
    email = EXCLUDED.email,
    password_hash = EXCLUDED.password_hash,
    updated_at = now();
