-- 000003_seed_procedures.up.sql
-- Seed procedures + ACTIVE versions from docs/phase-0/seeds/*.json
-- Requires 000002 (commune xa_chu_se + domains).

-- Seed admin (created_by for procedure_versions)
INSERT INTO users (id, role, full_name, email)
VALUES (
    'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee',
    'ADMIN',
    'Seed Admin',
    'admin@chuse.vn'
)
ON CONFLICT (id) DO NOTHING;

-- === chung_thuc_ban_sao ===
DO $$
DECLARE
  v_proc_id uuid;
  v_ver_id uuid;
BEGIN
  INSERT INTO procedures (procedure_code, domain_id, name, xa_id)
  VALUES ('chung_thuc_ban_sao', 'ho_tich_chung_thuc', 'Chứng thực bản sao', 'xa_chu_se')
  ON CONFLICT (xa_id, procedure_code) DO UPDATE
    SET name = EXCLUDED.name,
        domain_id = EXCLUDED.domain_id,
        updated_at = now()
  RETURNING id INTO v_proc_id;

  SELECT id INTO v_proc_id FROM procedures
  WHERE xa_id = 'xa_chu_se' AND procedure_code = 'chung_thuc_ban_sao';

  INSERT INTO procedure_versions (
    procedure_id, version, status, definition, source_draft_id, created_by, approved_by, approved_at
  )
  SELECT
    v_proc_id,
    '1.0.0',
    'ACTIVE',
    $def_chung_thuc_ban_sao${"procedure_code": "chung_thuc_ban_sao", "domain": "ho_tich_chung_thuc", "name": "Chứng thực bản sao", "description": "Hướng dẫn chứng thực bản sao từ bản chính tại UBND xã.", "xa_id": "xa_chu_se", "version": "1.0.0", "status": "ACTIVE", "authority_level": "xa", "intent_examples": ["Tôi muốn chứng thực bằng đại học", "Chứng thực bản sao CCCD", "Công chứng photo giấy tờ tại xã"], "slots": {"loai_giay_to": {"type": "string", "question": "Anh/chị cần chứng thực bản sao loại giấy tờ nào?", "description": "VD: bằng đại học, CCCD, giấy khai sinh..."}, "so_ban_sao": {"type": "number", "question": "Cần chứng thực bao nhiêu bản sao?"}}, "required_slots": [], "conditional_slots": [], "ask_policy": {"ask_mode": "all_missing", "completion_policy": "all_required_slots", "allow_direct_answer_if_no_required_slots": true}, "guidance": {"summary": "Để chứng thực bản sao, mang bản chính và bản photo đến bộ phận tiếp nhận của UBND xã.", "checklist": ["Bản chính giấy tờ cần chứng thực", "Bản photo/photocopy cần chứng thực", "CCCD của người yêu cầu chứng thực"], "where_to_submit": "Bộ phận tiếp nhận và trả kết quả — UBND xã", "notes": ["Một số loại giấy tờ có thể thuộc thẩm quyền chứng thực khác; nếu xã không chứng thực được sẽ hướng dẫn nơi phù hợp.", "Nên mang đủ số bản photo cần dùng để tránh đi lại.", "Seed tạm — cần đối chiếu quy định chứng thực hiện hành của xã."], "fee_hint": "Theo mức phí chứng thực bản sao do địa phương quy định.", "processing_time_hint": "Thường nhận kết quả nhanh trong buổi làm việc nếu hồ sơ hợp lệ."}, "citations": [{"doc_id": "seed_chung_thuc_ban_sao_v1", "title": "Hướng dẫn chứng thực bản sao (manual seed)", "source_type": "manual_seed", "effective_date": "2026-01-01", "issuer": "Seed Phase 0 — chờ văn bản xã thay thế"}], "metadata": {"priority": "P0", "preferred_route": "DIRECT_ANSWER", "seeded_at": "2026-08-04", "needs_official_pdf": true}}$def_chung_thuc_ban_sao$::jsonb,
    NULL,
    'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'::uuid,
    'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'::uuid,
    now()
  WHERE NOT EXISTS (
    SELECT 1 FROM procedure_versions pv
    WHERE pv.procedure_id = v_proc_id AND pv.version = '1.0.0'
  );

  SELECT id INTO v_ver_id FROM procedure_versions
  WHERE procedure_id = v_proc_id AND version = '1.0.0';

  UPDATE procedure_versions
  SET status = 'ACTIVE',
      definition = $def_chung_thuc_ban_sao${"procedure_code": "chung_thuc_ban_sao", "domain": "ho_tich_chung_thuc", "name": "Chứng thực bản sao", "description": "Hướng dẫn chứng thực bản sao từ bản chính tại UBND xã.", "xa_id": "xa_chu_se", "version": "1.0.0", "status": "ACTIVE", "authority_level": "xa", "intent_examples": ["Tôi muốn chứng thực bằng đại học", "Chứng thực bản sao CCCD", "Công chứng photo giấy tờ tại xã"], "slots": {"loai_giay_to": {"type": "string", "question": "Anh/chị cần chứng thực bản sao loại giấy tờ nào?", "description": "VD: bằng đại học, CCCD, giấy khai sinh..."}, "so_ban_sao": {"type": "number", "question": "Cần chứng thực bao nhiêu bản sao?"}}, "required_slots": [], "conditional_slots": [], "ask_policy": {"ask_mode": "all_missing", "completion_policy": "all_required_slots", "allow_direct_answer_if_no_required_slots": true}, "guidance": {"summary": "Để chứng thực bản sao, mang bản chính và bản photo đến bộ phận tiếp nhận của UBND xã.", "checklist": ["Bản chính giấy tờ cần chứng thực", "Bản photo/photocopy cần chứng thực", "CCCD của người yêu cầu chứng thực"], "where_to_submit": "Bộ phận tiếp nhận và trả kết quả — UBND xã", "notes": ["Một số loại giấy tờ có thể thuộc thẩm quyền chứng thực khác; nếu xã không chứng thực được sẽ hướng dẫn nơi phù hợp.", "Nên mang đủ số bản photo cần dùng để tránh đi lại.", "Seed tạm — cần đối chiếu quy định chứng thực hiện hành của xã."], "fee_hint": "Theo mức phí chứng thực bản sao do địa phương quy định.", "processing_time_hint": "Thường nhận kết quả nhanh trong buổi làm việc nếu hồ sơ hợp lệ."}, "citations": [{"doc_id": "seed_chung_thuc_ban_sao_v1", "title": "Hướng dẫn chứng thực bản sao (manual seed)", "source_type": "manual_seed", "effective_date": "2026-01-01", "issuer": "Seed Phase 0 — chờ văn bản xã thay thế"}], "metadata": {"priority": "P0", "preferred_route": "DIRECT_ANSWER", "seeded_at": "2026-08-04", "needs_official_pdf": true}}$def_chung_thuc_ban_sao$::jsonb,
      approved_by = 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'::uuid,
      approved_at = COALESCE(approved_at, now())
  WHERE id = v_ver_id;

  UPDATE procedures
  SET active_version_id = v_ver_id,
      updated_at = now()
  WHERE id = v_proc_id;
END $$;

-- === dk_bhyt_ho_gia_dinh ===
DO $$
DECLARE
  v_proc_id uuid;
  v_ver_id uuid;
BEGIN
  INSERT INTO procedures (procedure_code, domain_id, name, xa_id)
  VALUES ('dk_bhyt_ho_gia_dinh', 'bao_hiem_chinh_sach_xh', 'Đăng ký BHYT hộ gia đình', 'xa_chu_se')
  ON CONFLICT (xa_id, procedure_code) DO UPDATE
    SET name = EXCLUDED.name,
        domain_id = EXCLUDED.domain_id,
        updated_at = now()
  RETURNING id INTO v_proc_id;

  SELECT id INTO v_proc_id FROM procedures
  WHERE xa_id = 'xa_chu_se' AND procedure_code = 'dk_bhyt_ho_gia_dinh';

  INSERT INTO procedure_versions (
    procedure_id, version, status, definition, source_draft_id, created_by, approved_by, approved_at
  )
  SELECT
    v_proc_id,
    '1.0.0',
    'ACTIVE',
    $def_dk_bhyt_ho_gia_dinh${"procedure_code": "dk_bhyt_ho_gia_dinh", "domain": "bao_hiem_chinh_sach_xh", "name": "Đăng ký BHYT hộ gia đình", "description": "Hướng dẫn đăng ký tham gia BHYT theo hộ gia đình qua kênh xã/đại lý.", "xa_id": "xa_chu_se", "version": "1.0.0", "status": "ACTIVE", "authority_level": "xa", "intent_examples": ["Đăng ký BHYT hộ gia đình", "Mua bảo hiểm y tế cho cả nhà", "Làm BHYT theo hộ"], "slots": {"so_thanh_vien": {"type": "number", "question": "Hộ gia đình có bao nhiêu thành viên tham gia BHYT?"}, "da_co_so_ho_khau_hoac_cu_tru": {"type": "boolean", "question": "Các thành viên đã có thông tin hộ khẩu/cư trú hợp lệ chưa?"}, "co_nguoi_dang_tham_gia_noi_khac": {"type": "boolean", "question": "Trong hộ có ai đang tham gia BHYT ở nơi khác (cơ quan/trường học) không?"}}, "required_slots": ["so_thanh_vien", "da_co_so_ho_khau_hoac_cu_tru"], "conditional_slots": [], "ask_policy": {"ask_mode": "all_missing", "completion_policy": "all_required_slots", "allow_direct_answer_if_no_required_slots": true}, "guidance": {"summary": "Chuẩn bị danh sách thành viên và giấy tờ tùy thân để đăng ký BHYT hộ gia đình tại xã hoặc điểm thu được chỉ định.", "checklist": ["CCCD của chủ hộ và các thành viên tham gia", "Thông tin hộ khẩu/cư trú (hoặc dữ liệu dân cư tương đương)", "Tờ khai tham gia BHYT theo mẫu", "Tiền đóng BHYT theo mức quy định (sau khi được hướng dẫn mức đóng)"], "where_to_submit": "UBND xã / tổ chức thu BHYT được chỉ định tại địa bàn", "notes": ["Thành viên đã có BHYT nơi khác thường không đóng trùng theo hộ.", "Mức đóng và giảm trừ theo số thành viên tham gia — hỏi cán bộ để tính chính xác.", "Seed tạm — cần đối chiếu hướng dẫn BHXH/xã hiện hành trước prod."], "fee_hint": "Mức đóng theo % lương cơ sở và số thành viên tham gia (xác nhận tại điểm thu).", "processing_time_hint": "Thời gian cấp thẻ/mã BHYT theo quy trình BHXH; xã sẽ báo lịch trả."}, "citations": [{"doc_id": "seed_dk_bhyt_hgd_v1", "title": "Hướng dẫn đăng ký BHYT hộ gia đình (manual seed)", "source_type": "manual_seed", "effective_date": "2026-01-01", "issuer": "Seed Phase 0 — chờ văn bản xã/BHXH thay thế"}], "metadata": {"priority": "P0", "seeded_at": "2026-08-04", "needs_official_pdf": true}}$def_dk_bhyt_ho_gia_dinh$::jsonb,
    NULL,
    'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'::uuid,
    'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'::uuid,
    now()
  WHERE NOT EXISTS (
    SELECT 1 FROM procedure_versions pv
    WHERE pv.procedure_id = v_proc_id AND pv.version = '1.0.0'
  );

  SELECT id INTO v_ver_id FROM procedure_versions
  WHERE procedure_id = v_proc_id AND version = '1.0.0';

  UPDATE procedure_versions
  SET status = 'ACTIVE',
      definition = $def_dk_bhyt_ho_gia_dinh${"procedure_code": "dk_bhyt_ho_gia_dinh", "domain": "bao_hiem_chinh_sach_xh", "name": "Đăng ký BHYT hộ gia đình", "description": "Hướng dẫn đăng ký tham gia BHYT theo hộ gia đình qua kênh xã/đại lý.", "xa_id": "xa_chu_se", "version": "1.0.0", "status": "ACTIVE", "authority_level": "xa", "intent_examples": ["Đăng ký BHYT hộ gia đình", "Mua bảo hiểm y tế cho cả nhà", "Làm BHYT theo hộ"], "slots": {"so_thanh_vien": {"type": "number", "question": "Hộ gia đình có bao nhiêu thành viên tham gia BHYT?"}, "da_co_so_ho_khau_hoac_cu_tru": {"type": "boolean", "question": "Các thành viên đã có thông tin hộ khẩu/cư trú hợp lệ chưa?"}, "co_nguoi_dang_tham_gia_noi_khac": {"type": "boolean", "question": "Trong hộ có ai đang tham gia BHYT ở nơi khác (cơ quan/trường học) không?"}}, "required_slots": ["so_thanh_vien", "da_co_so_ho_khau_hoac_cu_tru"], "conditional_slots": [], "ask_policy": {"ask_mode": "all_missing", "completion_policy": "all_required_slots", "allow_direct_answer_if_no_required_slots": true}, "guidance": {"summary": "Chuẩn bị danh sách thành viên và giấy tờ tùy thân để đăng ký BHYT hộ gia đình tại xã hoặc điểm thu được chỉ định.", "checklist": ["CCCD của chủ hộ và các thành viên tham gia", "Thông tin hộ khẩu/cư trú (hoặc dữ liệu dân cư tương đương)", "Tờ khai tham gia BHYT theo mẫu", "Tiền đóng BHYT theo mức quy định (sau khi được hướng dẫn mức đóng)"], "where_to_submit": "UBND xã / tổ chức thu BHYT được chỉ định tại địa bàn", "notes": ["Thành viên đã có BHYT nơi khác thường không đóng trùng theo hộ.", "Mức đóng và giảm trừ theo số thành viên tham gia — hỏi cán bộ để tính chính xác.", "Seed tạm — cần đối chiếu hướng dẫn BHXH/xã hiện hành trước prod."], "fee_hint": "Mức đóng theo % lương cơ sở và số thành viên tham gia (xác nhận tại điểm thu).", "processing_time_hint": "Thời gian cấp thẻ/mã BHYT theo quy trình BHXH; xã sẽ báo lịch trả."}, "citations": [{"doc_id": "seed_dk_bhyt_hgd_v1", "title": "Hướng dẫn đăng ký BHYT hộ gia đình (manual seed)", "source_type": "manual_seed", "effective_date": "2026-01-01", "issuer": "Seed Phase 0 — chờ văn bản xã/BHXH thay thế"}], "metadata": {"priority": "P0", "seeded_at": "2026-08-04", "needs_official_pdf": true}}$def_dk_bhyt_ho_gia_dinh$::jsonb,
      approved_by = 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'::uuid,
      approved_at = COALESCE(approved_at, now())
  WHERE id = v_ver_id;

  UPDATE procedures
  SET active_version_id = v_ver_id,
      updated_at = now()
  WHERE id = v_proc_id;
END $$;

-- === dk_khai_sinh ===
DO $$
DECLARE
  v_proc_id uuid;
  v_ver_id uuid;
BEGIN
  INSERT INTO procedures (procedure_code, domain_id, name, xa_id)
  VALUES ('dk_khai_sinh', 'ho_tich_chung_thuc', 'Đăng ký khai sinh', 'xa_chu_se')
  ON CONFLICT (xa_id, procedure_code) DO UPDATE
    SET name = EXCLUDED.name,
        domain_id = EXCLUDED.domain_id,
        updated_at = now()
  RETURNING id INTO v_proc_id;

  SELECT id INTO v_proc_id FROM procedures
  WHERE xa_id = 'xa_chu_se' AND procedure_code = 'dk_khai_sinh';

  INSERT INTO procedure_versions (
    procedure_id, version, status, definition, source_draft_id, created_by, approved_by, approved_at
  )
  SELECT
    v_proc_id,
    '1.0.0',
    'ACTIVE',
    $def_dk_khai_sinh${"procedure_code": "dk_khai_sinh", "domain": "ho_tich_chung_thuc", "name": "Đăng ký khai sinh", "description": "Hướng dẫn chuẩn bị hồ sơ và nơi nộp đăng ký khai sinh tại xã.", "xa_id": "xa_chu_se", "version": "1.0.0", "status": "ACTIVE", "authority_level": "xa", "intent_examples": ["Tôi muốn làm giấy khai sinh cho con", "Đăng ký khai sinh", "Làm khai sinh cho bé mới sinh"], "slots": {"noi_sinh": {"type": "string", "question": "Bé sinh ở đâu (bệnh viện/cơ sở y tế hay tại nhà, thuộc xã/phường nào)?", "description": "Nơi sinh ảnh hưởng nơi nộp và giấy tờ thay thế"}, "da_ket_hon": {"type": "boolean", "question": "Cha mẹ bé đã đăng ký kết hôn chưa?"}, "co_giay_chung_sinh": {"type": "boolean", "question": "Anh/chị có giấy chứng sinh không?"}, "nguoi_di_dang_ky": {"type": "enum", "enum_values": ["cha", "me", "ong_ba", "nguoi_duoc_uy_quyen"], "question": "Ai sẽ đi đăng ký khai sinh?", "description": "Dùng để gợi ý giấy tờ kèm theo nếu không phải cha/mẹ"}}, "required_slots": ["noi_sinh", "da_ket_hon", "co_giay_chung_sinh"], "conditional_slots": [{"when": {"slot": "co_giay_chung_sinh", "equals": false}, "require": ["nguoi_di_dang_ky"]}], "ask_policy": {"ask_mode": "all_missing", "completion_policy": "all_required_slots", "allow_direct_answer_if_no_required_slots": true}, "guidance": {"summary": "Sau khi đủ thông tin, chuẩn bị hồ sơ đăng ký khai sinh và nộp tại UBND xã theo hướng dẫn địa phương.", "checklist": ["Giấy chứng sinh (nếu có) hoặc giấy tờ thay thế theo hướng dẫn xã", "CCCD/hộ chiếu của cha và/hoặc mẹ", "Giấy đăng ký kết hôn (nếu cha mẹ đã kết hôn)", "Tờ khai đăng ký khai sinh (mẫu tại bộ phận tiếp nhận)"], "where_to_submit": "Bộ phận tiếp nhận và trả kết quả — UBND xã (theo nơi cư trú của cha/mẹ hoặc nơi sinh, tùy trường hợp)", "notes": ["Nên đăng ký sớm trong thời hạn quy định.", "Nếu chưa có giấy chứng sinh, hỏi cán bộ hộ tịch giấy tờ thay thế được chấp nhận tại xã.", "Nội dung này là seed tạm; cần đối chiếu văn bản hiệu lực của xã trước khi đưa production."], "fee_hint": "Theo mức lệ phí hộ tịch do địa phương quy định (có thể miễn/giảm theo đối tượng).", "processing_time_hint": "Thường trả kết quả trong ngày làm việc nếu hồ sơ đầy đủ (xác nhận lại tại quầy)."}, "citations": [{"doc_id": "seed_dk_khai_sinh_v1", "title": "Hướng dẫn đăng ký khai sinh (manual seed)", "source_type": "manual_seed", "effective_date": "2026-01-01", "issuer": "Seed Phase 0 — chờ văn bản xã thay thế"}], "metadata": {"priority": "P0", "seeded_at": "2026-08-04", "needs_official_pdf": true}}$def_dk_khai_sinh$::jsonb,
    NULL,
    'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'::uuid,
    'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'::uuid,
    now()
  WHERE NOT EXISTS (
    SELECT 1 FROM procedure_versions pv
    WHERE pv.procedure_id = v_proc_id AND pv.version = '1.0.0'
  );

  SELECT id INTO v_ver_id FROM procedure_versions
  WHERE procedure_id = v_proc_id AND version = '1.0.0';

  UPDATE procedure_versions
  SET status = 'ACTIVE',
      definition = $def_dk_khai_sinh${"procedure_code": "dk_khai_sinh", "domain": "ho_tich_chung_thuc", "name": "Đăng ký khai sinh", "description": "Hướng dẫn chuẩn bị hồ sơ và nơi nộp đăng ký khai sinh tại xã.", "xa_id": "xa_chu_se", "version": "1.0.0", "status": "ACTIVE", "authority_level": "xa", "intent_examples": ["Tôi muốn làm giấy khai sinh cho con", "Đăng ký khai sinh", "Làm khai sinh cho bé mới sinh"], "slots": {"noi_sinh": {"type": "string", "question": "Bé sinh ở đâu (bệnh viện/cơ sở y tế hay tại nhà, thuộc xã/phường nào)?", "description": "Nơi sinh ảnh hưởng nơi nộp và giấy tờ thay thế"}, "da_ket_hon": {"type": "boolean", "question": "Cha mẹ bé đã đăng ký kết hôn chưa?"}, "co_giay_chung_sinh": {"type": "boolean", "question": "Anh/chị có giấy chứng sinh không?"}, "nguoi_di_dang_ky": {"type": "enum", "enum_values": ["cha", "me", "ong_ba", "nguoi_duoc_uy_quyen"], "question": "Ai sẽ đi đăng ký khai sinh?", "description": "Dùng để gợi ý giấy tờ kèm theo nếu không phải cha/mẹ"}}, "required_slots": ["noi_sinh", "da_ket_hon", "co_giay_chung_sinh"], "conditional_slots": [{"when": {"slot": "co_giay_chung_sinh", "equals": false}, "require": ["nguoi_di_dang_ky"]}], "ask_policy": {"ask_mode": "all_missing", "completion_policy": "all_required_slots", "allow_direct_answer_if_no_required_slots": true}, "guidance": {"summary": "Sau khi đủ thông tin, chuẩn bị hồ sơ đăng ký khai sinh và nộp tại UBND xã theo hướng dẫn địa phương.", "checklist": ["Giấy chứng sinh (nếu có) hoặc giấy tờ thay thế theo hướng dẫn xã", "CCCD/hộ chiếu của cha và/hoặc mẹ", "Giấy đăng ký kết hôn (nếu cha mẹ đã kết hôn)", "Tờ khai đăng ký khai sinh (mẫu tại bộ phận tiếp nhận)"], "where_to_submit": "Bộ phận tiếp nhận và trả kết quả — UBND xã (theo nơi cư trú của cha/mẹ hoặc nơi sinh, tùy trường hợp)", "notes": ["Nên đăng ký sớm trong thời hạn quy định.", "Nếu chưa có giấy chứng sinh, hỏi cán bộ hộ tịch giấy tờ thay thế được chấp nhận tại xã.", "Nội dung này là seed tạm; cần đối chiếu văn bản hiệu lực của xã trước khi đưa production."], "fee_hint": "Theo mức lệ phí hộ tịch do địa phương quy định (có thể miễn/giảm theo đối tượng).", "processing_time_hint": "Thường trả kết quả trong ngày làm việc nếu hồ sơ đầy đủ (xác nhận lại tại quầy)."}, "citations": [{"doc_id": "seed_dk_khai_sinh_v1", "title": "Hướng dẫn đăng ký khai sinh (manual seed)", "source_type": "manual_seed", "effective_date": "2026-01-01", "issuer": "Seed Phase 0 — chờ văn bản xã thay thế"}], "metadata": {"priority": "P0", "seeded_at": "2026-08-04", "needs_official_pdf": true}}$def_dk_khai_sinh$::jsonb,
      approved_by = 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'::uuid,
      approved_at = COALESCE(approved_at, now())
  WHERE id = v_ver_id;

  UPDATE procedures
  SET active_version_id = v_ver_id,
      updated_at = now()
  WHERE id = v_proc_id;
END $$;

-- === tra_cuu_quy_hoach ===
DO $$
DECLARE
  v_proc_id uuid;
  v_ver_id uuid;
BEGIN
  INSERT INTO procedures (procedure_code, domain_id, name, xa_id)
  VALUES ('tra_cuu_quy_hoach', 'dat_dai_nha_o_quy_hoach', 'Tra cứu quy hoạch', 'xa_chu_se')
  ON CONFLICT (xa_id, procedure_code) DO UPDATE
    SET name = EXCLUDED.name,
        domain_id = EXCLUDED.domain_id,
        updated_at = now()
  RETURNING id INTO v_proc_id;

  SELECT id INTO v_proc_id FROM procedures
  WHERE xa_id = 'xa_chu_se' AND procedure_code = 'tra_cuu_quy_hoach';

  INSERT INTO procedure_versions (
    procedure_id, version, status, definition, source_draft_id, created_by, approved_by, approved_at
  )
  SELECT
    v_proc_id,
    '1.0.0',
    'ACTIVE',
    $def_tra_cuu_quy_hoach${"procedure_code": "tra_cuu_quy_hoach", "domain": "dat_dai_nha_o_quy_hoach", "name": "Tra cứu quy hoạch", "description": "Hướng dẫn cách tra cứu thông tin quy hoạch tại địa bàn xã.", "xa_id": "xa_chu_se", "version": "1.0.0", "status": "ACTIVE", "authority_level": "mixed", "intent_examples": ["Tra cứu quy hoạch thửa đất", "Đất tôi có nằm trong quy hoạch không", "Xem quy hoạch sử dụng đất xã"], "slots": {"vi_tri_thua_dat": {"type": "string", "question": "Anh/chị muốn tra cứu thửa đất ở vị trí nào (địa chỉ/ấp/thôn hoặc số tờ/số thửa nếu có)?"}, "muc_dich_tra_cuu": {"type": "enum", "enum_values": ["xay_nha", "chuyen_nhuong", "tim_hieu_chung"], "question": "Mục đích tra cứu là gì (xây nhà, chuyển nhượng, hay tìm hiểu chung)?"}}, "required_slots": ["vi_tri_thua_dat"], "conditional_slots": [], "ask_policy": {"ask_mode": "all_missing", "completion_policy": "all_required_slots", "allow_direct_answer_if_no_required_slots": true}, "guidance": {"summary": "Cung cấp vị trí thửa đất để bộ phận chuyên môn hướng dẫn tra cứu quy hoạch theo bản đồ/hồ sơ địa phương.", "checklist": ["Thông tin vị trí thửa đất (địa chỉ hoặc số tờ/số thửa)", "Giấy chứng nhận quyền sử dụng đất (nếu có) để đối chiếu", "CCCD người yêu cầu tra cứu"], "where_to_submit": "Bộ phận địa chính / tiếp nhận UBND xã (có thể hướng dẫn tra cứu tại cổng thông tin hoặc cấp huyện)", "notes": ["Kết quả tra cứu mang tính hướng dẫn; xác nhận chính thức theo hồ sơ quy hoạch được công bố.", "Seed tạm — cần gắn nguồn bản đồ/quy hoạch chính thức của xã."], "fee_hint": "Theo quy định cung cấp thông tin của địa phương (một số trường hợp miễn phí tra cứu cơ bản).", "processing_time_hint": "Tùy hình thức tra cứu tại quầy hoặc trực tuyến."}, "citations": [{"doc_id": "seed_tra_cuu_quy_hoach_v1", "title": "Hướng dẫn tra cứu quy hoạch (manual seed)", "source_type": "manual_seed", "effective_date": "2026-01-01", "issuer": "Seed Phase 0 — chờ nguồn quy hoạch xã"}], "metadata": {"priority": "P0", "seeded_at": "2026-08-04", "needs_official_pdf": true}}$def_tra_cuu_quy_hoach$::jsonb,
    NULL,
    'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'::uuid,
    'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'::uuid,
    now()
  WHERE NOT EXISTS (
    SELECT 1 FROM procedure_versions pv
    WHERE pv.procedure_id = v_proc_id AND pv.version = '1.0.0'
  );

  SELECT id INTO v_ver_id FROM procedure_versions
  WHERE procedure_id = v_proc_id AND version = '1.0.0';

  UPDATE procedure_versions
  SET status = 'ACTIVE',
      definition = $def_tra_cuu_quy_hoach${"procedure_code": "tra_cuu_quy_hoach", "domain": "dat_dai_nha_o_quy_hoach", "name": "Tra cứu quy hoạch", "description": "Hướng dẫn cách tra cứu thông tin quy hoạch tại địa bàn xã.", "xa_id": "xa_chu_se", "version": "1.0.0", "status": "ACTIVE", "authority_level": "mixed", "intent_examples": ["Tra cứu quy hoạch thửa đất", "Đất tôi có nằm trong quy hoạch không", "Xem quy hoạch sử dụng đất xã"], "slots": {"vi_tri_thua_dat": {"type": "string", "question": "Anh/chị muốn tra cứu thửa đất ở vị trí nào (địa chỉ/ấp/thôn hoặc số tờ/số thửa nếu có)?"}, "muc_dich_tra_cuu": {"type": "enum", "enum_values": ["xay_nha", "chuyen_nhuong", "tim_hieu_chung"], "question": "Mục đích tra cứu là gì (xây nhà, chuyển nhượng, hay tìm hiểu chung)?"}}, "required_slots": ["vi_tri_thua_dat"], "conditional_slots": [], "ask_policy": {"ask_mode": "all_missing", "completion_policy": "all_required_slots", "allow_direct_answer_if_no_required_slots": true}, "guidance": {"summary": "Cung cấp vị trí thửa đất để bộ phận chuyên môn hướng dẫn tra cứu quy hoạch theo bản đồ/hồ sơ địa phương.", "checklist": ["Thông tin vị trí thửa đất (địa chỉ hoặc số tờ/số thửa)", "Giấy chứng nhận quyền sử dụng đất (nếu có) để đối chiếu", "CCCD người yêu cầu tra cứu"], "where_to_submit": "Bộ phận địa chính / tiếp nhận UBND xã (có thể hướng dẫn tra cứu tại cổng thông tin hoặc cấp huyện)", "notes": ["Kết quả tra cứu mang tính hướng dẫn; xác nhận chính thức theo hồ sơ quy hoạch được công bố.", "Seed tạm — cần gắn nguồn bản đồ/quy hoạch chính thức của xã."], "fee_hint": "Theo quy định cung cấp thông tin của địa phương (một số trường hợp miễn phí tra cứu cơ bản).", "processing_time_hint": "Tùy hình thức tra cứu tại quầy hoặc trực tuyến."}, "citations": [{"doc_id": "seed_tra_cuu_quy_hoach_v1", "title": "Hướng dẫn tra cứu quy hoạch (manual seed)", "source_type": "manual_seed", "effective_date": "2026-01-01", "issuer": "Seed Phase 0 — chờ nguồn quy hoạch xã"}], "metadata": {"priority": "P0", "seeded_at": "2026-08-04", "needs_official_pdf": true}}$def_tra_cuu_quy_hoach$::jsonb,
      approved_by = 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'::uuid,
      approved_at = COALESCE(approved_at, now())
  WHERE id = v_ver_id;

  UPDATE procedures
  SET active_version_id = v_ver_id,
      updated_at = now()
  WHERE id = v_proc_id;
END $$;

-- === xin_giay_phep_xay_dung ===
DO $$
DECLARE
  v_proc_id uuid;
  v_ver_id uuid;
BEGIN
  INSERT INTO procedures (procedure_code, domain_id, name, xa_id)
  VALUES ('xin_giay_phep_xay_dung', 'dat_dai_nha_o_quy_hoach', 'Xin giấy phép xây dựng', 'xa_chu_se')
  ON CONFLICT (xa_id, procedure_code) DO UPDATE
    SET name = EXCLUDED.name,
        domain_id = EXCLUDED.domain_id,
        updated_at = now()
  RETURNING id INTO v_proc_id;

  SELECT id INTO v_proc_id FROM procedures
  WHERE xa_id = 'xa_chu_se' AND procedure_code = 'xin_giay_phep_xay_dung';

  INSERT INTO procedure_versions (
    procedure_id, version, status, definition, source_draft_id, created_by, approved_by, approved_at
  )
  SELECT
    v_proc_id,
    '1.0.0',
    'ACTIVE',
    $def_xin_giay_phep_xay_dung${"procedure_code": "xin_giay_phep_xay_dung", "domain": "dat_dai_nha_o_quy_hoach", "name": "Xin giấy phép xây dựng", "description": "Hướng dẫn chuẩn bị hồ sơ xin giấy phép xây dựng; thẩm quyền có thể ở xã hoặc cấp trên tùy quy mô.", "xa_id": "xa_chu_se", "version": "1.0.0", "status": "ACTIVE", "authority_level": "mixed", "intent_examples": ["Xin giấy phép xây dựng nhà", "Làm GPXD", "Muốn xây nhà cần giấy phép gì"], "slots": {"loai_cong_trinh": {"type": "enum", "enum_values": ["nha_o_rieng_le", "cong_trinh_khac"], "question": "Công trình xin phép thuộc loại nào (nhà ở riêng lẻ hay công trình khác)?"}, "co_dat_dung_ten": {"type": "boolean", "question": "Thửa đất xây dựng đã có giấy chứng nhận đứng tên người xin phép chưa?"}, "nam_trong_khu_can_phep": {"type": "boolean", "question": "Khu vực xây dựng có thuộc trường hợp bắt buộc xin giấy phép theo hướng dẫn địa phương không? (nếu chưa rõ chọn chưa rõ và nhờ cán bộ xác nhận)"}, "dien_tich_xay_dung": {"type": "string", "question": "Diện tích xây dựng dự kiến khoảng bao nhiêu (m2)?"}}, "required_slots": ["loai_cong_trinh", "co_dat_dung_ten"], "conditional_slots": [{"when": {"slot": "loai_cong_trinh", "equals": "nha_o_rieng_le"}, "require": ["dien_tich_xay_dung"]}], "ask_policy": {"ask_mode": "all_missing", "completion_policy": "all_required_slots", "allow_direct_answer_if_no_required_slots": true}, "guidance": {"summary": "Chuẩn bị hồ sơ xin giấy phép xây dựng theo loại công trình và nộp tại bộ phận tiếp nhận; một số trường hợp xã tiếp nhận rồi chuyển cấp huyện.", "checklist": ["Đơn đề nghị cấp giấy phép xây dựng", "Giấy tờ về quyền sử dụng đất (GCN hoặc giấy tờ hợp pháp tương đương)", "Bản vẽ thiết kế xây dựng theo quy định", "Giấy tờ tùy thân của chủ đầu tư/người được ủy quyền"], "where_to_submit": "Bộ phận tiếp nhận UBND xã (tiếp nhận/hướng dẫn; thẩm quyền cấp phép có thể thuộc huyện tùy quy mô)", "notes": ["Không phải mọi trường hợp nhà ở riêng lẻ đều phải xin GPXD — cần xác nhận theo quy hoạch/địa bàn xã.", "Nếu đất chưa đúng tên hoặc tranh chấp, hồ sơ có thể bị từ chối/yêu cầu bổ sung.", "Seed tạm — bắt buộc đối chiếu quy định quy hoạch và thẩm quyền địa phương trước prod."], "fee_hint": "Theo mức phí/lệ phí cấp phép xây dựng hiện hành.", "processing_time_hint": "Thời hạn giải quyết theo quy định từng cấp thẩm quyền; hỏi quầy để biết ngày hẹn trả."}, "citations": [{"doc_id": "seed_xin_gpxd_v1", "title": "Hướng dẫn xin giấy phép xây dựng (manual seed)", "source_type": "manual_seed", "effective_date": "2026-01-01", "issuer": "Seed Phase 0 — chờ văn bản xã/huyện thay thế"}], "metadata": {"priority": "P0", "seeded_at": "2026-08-04", "needs_official_pdf": true}}$def_xin_giay_phep_xay_dung$::jsonb,
    NULL,
    'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'::uuid,
    'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'::uuid,
    now()
  WHERE NOT EXISTS (
    SELECT 1 FROM procedure_versions pv
    WHERE pv.procedure_id = v_proc_id AND pv.version = '1.0.0'
  );

  SELECT id INTO v_ver_id FROM procedure_versions
  WHERE procedure_id = v_proc_id AND version = '1.0.0';

  UPDATE procedure_versions
  SET status = 'ACTIVE',
      definition = $def_xin_giay_phep_xay_dung${"procedure_code": "xin_giay_phep_xay_dung", "domain": "dat_dai_nha_o_quy_hoach", "name": "Xin giấy phép xây dựng", "description": "Hướng dẫn chuẩn bị hồ sơ xin giấy phép xây dựng; thẩm quyền có thể ở xã hoặc cấp trên tùy quy mô.", "xa_id": "xa_chu_se", "version": "1.0.0", "status": "ACTIVE", "authority_level": "mixed", "intent_examples": ["Xin giấy phép xây dựng nhà", "Làm GPXD", "Muốn xây nhà cần giấy phép gì"], "slots": {"loai_cong_trinh": {"type": "enum", "enum_values": ["nha_o_rieng_le", "cong_trinh_khac"], "question": "Công trình xin phép thuộc loại nào (nhà ở riêng lẻ hay công trình khác)?"}, "co_dat_dung_ten": {"type": "boolean", "question": "Thửa đất xây dựng đã có giấy chứng nhận đứng tên người xin phép chưa?"}, "nam_trong_khu_can_phep": {"type": "boolean", "question": "Khu vực xây dựng có thuộc trường hợp bắt buộc xin giấy phép theo hướng dẫn địa phương không? (nếu chưa rõ chọn chưa rõ và nhờ cán bộ xác nhận)"}, "dien_tich_xay_dung": {"type": "string", "question": "Diện tích xây dựng dự kiến khoảng bao nhiêu (m2)?"}}, "required_slots": ["loai_cong_trinh", "co_dat_dung_ten"], "conditional_slots": [{"when": {"slot": "loai_cong_trinh", "equals": "nha_o_rieng_le"}, "require": ["dien_tich_xay_dung"]}], "ask_policy": {"ask_mode": "all_missing", "completion_policy": "all_required_slots", "allow_direct_answer_if_no_required_slots": true}, "guidance": {"summary": "Chuẩn bị hồ sơ xin giấy phép xây dựng theo loại công trình và nộp tại bộ phận tiếp nhận; một số trường hợp xã tiếp nhận rồi chuyển cấp huyện.", "checklist": ["Đơn đề nghị cấp giấy phép xây dựng", "Giấy tờ về quyền sử dụng đất (GCN hoặc giấy tờ hợp pháp tương đương)", "Bản vẽ thiết kế xây dựng theo quy định", "Giấy tờ tùy thân của chủ đầu tư/người được ủy quyền"], "where_to_submit": "Bộ phận tiếp nhận UBND xã (tiếp nhận/hướng dẫn; thẩm quyền cấp phép có thể thuộc huyện tùy quy mô)", "notes": ["Không phải mọi trường hợp nhà ở riêng lẻ đều phải xin GPXD — cần xác nhận theo quy hoạch/địa bàn xã.", "Nếu đất chưa đúng tên hoặc tranh chấp, hồ sơ có thể bị từ chối/yêu cầu bổ sung.", "Seed tạm — bắt buộc đối chiếu quy định quy hoạch và thẩm quyền địa phương trước prod."], "fee_hint": "Theo mức phí/lệ phí cấp phép xây dựng hiện hành.", "processing_time_hint": "Thời hạn giải quyết theo quy định từng cấp thẩm quyền; hỏi quầy để biết ngày hẹn trả."}, "citations": [{"doc_id": "seed_xin_gpxd_v1", "title": "Hướng dẫn xin giấy phép xây dựng (manual seed)", "source_type": "manual_seed", "effective_date": "2026-01-01", "issuer": "Seed Phase 0 — chờ văn bản xã/huyện thay thế"}], "metadata": {"priority": "P0", "seeded_at": "2026-08-04", "needs_official_pdf": true}}$def_xin_giay_phep_xay_dung$::jsonb,
      approved_by = 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'::uuid,
      approved_at = COALESCE(approved_at, now())
  WHERE id = v_ver_id;

  UPDATE procedures
  SET active_version_id = v_ver_id,
      updated_at = now()
  WHERE id = v_proc_id;
END $$;

