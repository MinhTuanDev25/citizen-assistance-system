-- 000003_seed_procedures.down.sql
UPDATE procedures SET active_version_id = NULL
WHERE xa_id = 'xa_chu_se'
  AND procedure_code IN ('chung_thuc_ban_sao', 'dk_bhyt_ho_gia_dinh', 'dk_khai_sinh', 'tra_cuu_quy_hoach', 'xin_giay_phep_xay_dung');

DELETE FROM procedure_versions pv
USING procedures p
WHERE pv.procedure_id = p.id
  AND p.xa_id = 'xa_chu_se'
  AND p.procedure_code IN ('chung_thuc_ban_sao', 'dk_bhyt_ho_gia_dinh', 'dk_khai_sinh', 'tra_cuu_quy_hoach', 'xin_giay_phep_xay_dung');

DELETE FROM procedures
WHERE xa_id = 'xa_chu_se'
  AND procedure_code IN ('chung_thuc_ban_sao', 'dk_bhyt_ho_gia_dinh', 'dk_khai_sinh', 'tra_cuu_quy_hoach', 'xin_giay_phep_xay_dung');

DELETE FROM users WHERE id = 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee';
