-- 000002_seed_master.down.sql

DELETE FROM domains
WHERE id IN (
    'ho_tich_chung_thuc',
    'dat_dai_nha_o_quy_hoach',
    'bao_hiem_chinh_sach_xh'
);

DELETE FROM communes WHERE id = 'xa_chu_se';
