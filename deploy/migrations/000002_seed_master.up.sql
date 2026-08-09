-- 000002_seed_master.up.sql
-- Master data only (1 commune + 3 domains). Procedure JSON seeds load in Phase 1 app.

INSERT INTO communes (id, name, description, is_active)
VALUES (
    'xa_chu_se',
    'Chư Sê',
    'Huyện Chư Sê, tỉnh Gia Lai — phạm vi triển khai V1',
    true
)
ON CONFLICT (id) DO NOTHING;

INSERT INTO domains (id, name, description, sort_order, is_active)
VALUES
    ('ho_tich_chung_thuc', 'Hộ tịch & Chứng thực', NULL, 1, true),
    ('dat_dai_nha_o_quy_hoach', 'Đất đai, Nhà ở & Quy hoạch', NULL, 2, true),
    ('bao_hiem_chinh_sach_xh', 'Bảo hiểm & Chính sách xã hội', NULL, 3, true)
ON CONFLICT (id) DO NOTHING;
