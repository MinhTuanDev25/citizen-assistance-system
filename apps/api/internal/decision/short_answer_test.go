package decision

import "testing"

// The Admin-drafted "Đăng ký khai sinh" 1.0.1 slots.
func draftedKhaiSinh(t *testing.T) Definition {
	t.Helper()
	def, err := ParseDefinition([]byte(`{
		"procedure_code": "khai_sinh_v1", "name": "Đăng ký khai sinh",
		"slots": {
			"co_uy_quyen": {"type": "boolean", "question": "Việc đăng ký khai sinh có thực hiện theo văn bản ủy quyền không?"},
			"hinh_thuc_nop": {"type": "enum", "question": "Bạn muốn lựa chọn hình thức nộp hồ sơ nào?", "enum_values": ["truc_tiep", "truc_tuyen", "buu_chinh"], "enum_labels": ["Trực tiếp tại Trung tâm Phục vụ hành chính công", "Trực tuyến qua Cổng dịch vụ công quốc gia/VNeID", "Qua hệ thống bưu chính"]},
			"truong_hop_tre": {"type": "enum", "question": "Trẻ thuộc trường hợp nào sau đây?", "enum_values": ["sinh_thuong", "bi_bo_roi", "mang_thai_ho"], "enum_labels": ["Trẻ sinh thông thường", "Trẻ bị bỏ rơi", "Trẻ sinh ra do mang thai hộ"]},
			"tinh_trang_giay_chung_sinh": {"type": "enum", "question": "Bạn có Giấy chứng sinh hay giấy tờ thay thế nào?", "enum_values": ["co_giay_chung_sinh", "co_nguoi_lam_chung", "cam_doan"], "enum_labels": ["Có Giấy chứng sinh", "Không có Giấy chứng sinh, có văn bản người làm chứng", "Không có Giấy chứng sinh và không có người làm chứng, có giấy cam đoan"]}
		},
		"required_slots": ["hinh_thuc_nop", "truong_hop_tre", "co_uy_quyen"],
		"conditional_slots": [{"when": {"slot": "truong_hop_tre", "equals": "sinh_thuong"}, "require": ["tinh_trang_giay_chung_sinh"]}],
		"guidance": {"summary": "s", "checklist": ["Tờ khai"]}
	}`))
	if err != nil {
		t.Fatal(err)
	}
	return def
}

func TestShortAnswersToTheOnlyOpenChoice(t *testing.T) {
	def := draftedKhaiSinh(t)
	known := func(kv map[string]any) map[string]SlotValue {
		out := map[string]SlotValue{}
		for k, v := range kv {
			out[k] = SlotValue{Value: v, Status: StatusKnown}
		}
		return out
	}
	tre := map[string]any{"hinh_thuc_nop": "truc_tuyen", "co_uy_quyen": true}
	giay := map[string]any{"hinh_thuc_nop": "truc_tuyen", "co_uy_quyen": true, "truong_hop_tre": "sinh_thuong"}
	cases := []struct {
		prior    map[string]any
		msg, key string
		want     any // nil: must stay open
	}{
		{tre, "thông thường", "truong_hop_tre", "sinh_thuong"},
		{tre, "con tôi đẻ thường", "truong_hop_tre", "sinh_thuong"},
		{tre, "bỏ rơi", "truong_hop_tre", "bi_bo_roi"},
		{giay, "có", "tinh_trang_giay_chung_sinh", "co_giay_chung_sinh"},
		{giay, "có giấy", "tinh_trang_giay_chung_sinh", "co_giay_chung_sinh"},
		{giay, "dạ có ạ", "tinh_trang_giay_chung_sinh", "co_giay_chung_sinh"},
		{giay, "có người làm chứng", "tinh_trang_giay_chung_sinh", "co_nguoi_lam_chung"},
		{giay, "cam đoan", "tinh_trang_giay_chung_sinh", "cam_doan"},
		// A choice the citizen leans to is taken; see grounding's leaning-choice.
		{map[string]any{}, "chắc là tôi muốn trực tiếp luôn", "hinh_thuc_nop", "truc_tiep"},
		{map[string]any{}, "có lẽ nộp trực tuyến", "hinh_thuc_nop", "truc_tuyen"},
		// Ambiguous, hedged or unrelated: ask again.
		{giay, "không", "tinh_trang_giay_chung_sinh", nil},
		{giay, "không có giấy", "tinh_trang_giay_chung_sinh", nil},
		{giay, "chắc là có", "tinh_trang_giay_chung_sinh", nil},
		{map[string]any{}, "tôi chưa biết nộp trực tuyến hay trực tiếp", "hinh_thuc_nop", nil},
		{map[string]any{}, "không muốn nộp trực tuyến", "hinh_thuc_nop", nil},
		{tre, "bình thường hay bỏ rơi", "truong_hop_tre", nil},
		{tre, "xin chào", "truong_hop_tre", nil},
	}
	for _, tc := range cases {
		out := Decide(TurnInput{Definition: def, Prior: known(tc.prior), HadPriorState: true, Message: tc.msg})
		got := out.SlotState[tc.key]
		if tc.want == nil {
			if got.Filled() {
				t.Errorf("%q must leave %s open, got %v", tc.msg, tc.key, got.Value)
			}
			continue
		}
		if got.Value != tc.want {
			t.Errorf("%q: %s=%v, want %v", tc.msg, tc.key, got.Value, tc.want)
		}
	}
}
