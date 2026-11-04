package aiextract

import "testing"

// Slots of the Admin-drafted "Đăng ký khai sinh" 1.0.1 (no hand-written cue table).
var (
	draftedHinhThuc = SlotSpec{Type: "enum", Question: "Bạn muốn lựa chọn hình thức nộp hồ sơ nào?",
		EnumValues: []string{"truc_tiep", "truc_tuyen", "buu_chinh"},
		EnumLabels: []string{"Trực tiếp tại Trung tâm Phục vụ hành chính công", "Trực tuyến qua Cổng dịch vụ công quốc gia/VNeID", "Qua hệ thống bưu chính"}}
	draftedTruongHop = SlotSpec{Type: "enum", Question: "Trẻ thuộc trường hợp nào sau đây?",
		EnumValues: []string{"sinh_thuong", "bi_bo_roi", "mang_thai_ho"},
		EnumLabels: []string{"Trẻ sinh thông thường", "Trẻ bị bỏ rơi", "Trẻ sinh ra do mang thai hộ"}}
	draftedUyQuyen = SlotSpec{Type: "boolean", Question: "Việc đăng ký khai sinh có thực hiện theo văn bản ủy quyền không?"}
	draftedGiayCS  = SlotSpec{Type: "enum", Question: "Bạn có Giấy chứng sinh hay giấy tờ thay thế nào?",
		EnumValues: []string{"co_giay_chung_sinh", "co_nguoi_lam_chung", "cam_doan"},
		EnumLabels: []string{"Có Giấy chứng sinh", "Không có Giấy chứng sinh, có văn bản người làm chứng", "Không có Giấy chứng sinh và không có người làm chứng, có giấy cam đoan"}}
)

func TestNaturalAnswersToDraftedQuestions(t *testing.T) {
	const name = "Đăng ký khai sinh"
	const multi = "tôi muốn nộp trực tuyến, con tôi đẻ thường, có uỷ quyền"
	cases := []struct {
		id, msg, key string
		spec         SlotSpec
		value        any
		evidence     string
		accept       bool
	}{
		// What citizens actually typed.
		{"wish-is-the-answer", multi, "hinh_thuc_nop", draftedHinhThuc, "truc_tuyen", "tôi muốn nộp trực tuyến", true},
		{"filler-around-answer", multi, "truong_hop_tre", draftedTruongHop, "sinh_thuong", "con tôi đẻ thường", true},
		{"short-yes-boolean", multi, "co_uy_quyen", draftedUyQuyen, true, "có uỷ quyền", true},
		{"bare-co", "có", "tinh_trang_giay_chung_sinh", draftedGiayCS, "co_giay_chung_sinh", "có", true},
		{"co-giay", "có giấy", "tinh_trang_giay_chung_sinh", draftedGiayCS, "co_giay_chung_sinh", "có giấy", true},
		{"da-co-a", "dạ có ạ", "tinh_trang_giay_chung_sinh", draftedGiayCS, "co_giay_chung_sinh", "dạ có ạ", true},
		{"short-no-boolean", "không uỷ quyền", "co_uy_quyen", draftedUyQuyen, false, "không uỷ quyền", true},
		// A soft hedge leans to a choice the citizen owns: taken.
		{"leaning-choice", "chắc là tôi muốn trực tiếp luôn", "hinh_thuc_nop", draftedHinhThuc, "truc_tiep", "chắc là tôi muốn trực tiếp luôn", true},
		{"leaning-choice-short-span", "chắc là tôi muốn trực tiếp luôn", "hinh_thuc_nop", draftedHinhThuc, "truc_tiep", "trực tiếp", true},
		{"leaning-choice-will", "chắc là nộp trực tuyến", "hinh_thuc_nop", draftedHinhThuc, "truc_tuyen", "nộp trực tuyến", true},
		{"co-le-choice", "có lẽ tôi sẽ nộp qua bưu chính", "hinh_thuc_nop", draftedHinhThuc, "buu_chinh", "nộp qua bưu chính", true},

		// Still refused.
		{"refused-wish", "tôi không muốn nộp trực tuyến", "hinh_thuc_nop", draftedHinhThuc, "truc_tuyen", "nộp trực tuyến", false},
		{"undecided-choice", "tôi chưa biết có nộp trực tuyến được không", "hinh_thuc_nop", draftedHinhThuc, "truc_tuyen", "nộp trực tuyến", false},
		{"not-sure-choice", "tôi chưa chắc nộp trực tuyến", "hinh_thuc_nop", draftedHinhThuc, "truc_tuyen", "nộp trực tuyến", false},
		{"leaning-two-choices", "chắc là trực tuyến hoặc bưu chính", "hinh_thuc_nop", draftedHinhThuc, "truc_tuyen", "trực tuyến", false},
		{"hedged-fact-enum", "chắc là có giấy", "tinh_trang_giay_chung_sinh", draftedGiayCS, "co_giay_chung_sinh", "có giấy", false},
		{"two-choices", "nộp trực tuyến hay bưu chính", "hinh_thuc_nop", draftedHinhThuc, "truc_tuyen", "trực tuyến", false},
		{"khong-co-giay-ambiguous", "không có giấy", "tinh_trang_giay_chung_sinh", draftedGiayCS, "co_nguoi_lam_chung", "không có giấy", false},
		{"khong-co-giay-not-yes", "không có giấy", "tinh_trang_giay_chung_sinh", draftedGiayCS, "co_giay_chung_sinh", "không có giấy", false},
		{"co-from-another-answer", multi, "tinh_trang_giay_chung_sinh", draftedGiayCS, "co_giay_chung_sinh", "có", false},
		{"invented-span", "có", "tinh_trang_giay_chung_sinh", draftedGiayCS, "co_giay_chung_sinh", "có giấy chứng sinh", false},
		{"polarity-flip", "có uỷ quyền", "co_uy_quyen", draftedUyQuyen, false, "có uỷ quyền", false},
		{"hedged-boolean", "hình như có uỷ quyền", "co_uy_quyen", draftedUyQuyen, true, "hình như có uỷ quyền", false},
	}
	for _, tc := range cases {
		ok, reason := GroundSlotFor(name, tc.msg, tc.spec, tc.key, tc.value, tc.evidence)
		if ok != tc.accept {
			t.Errorf("%s: %q %s=%v ev=%q → ok=%v (reason=%s), want %v", tc.id, tc.msg, tc.key, tc.value, tc.evidence, ok, reason, tc.accept)
		}
	}
}
