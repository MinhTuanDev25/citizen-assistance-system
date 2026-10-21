package aiextract

import "testing"

func spec(key string) SlotSpec {
	switch key {
	case "da_ket_hon", "co_giay_chung_sinh":
		return SlotSpec{Type: "boolean"}
	case "nguoi_di_dang_ky":
		return SlotSpec{Type: "enum", EnumValues: []string{"cha", "me", "ong_ba", "nguoi_duoc_uy_quyen"}}
	default:
		return SlotSpec{Type: "string"}
	}
}

func TestH3GroundingMatrix(t *testing.T) {
	type row struct {
		id, message, key string
		value            any
		evidence         string
		accept           bool
	}
	cases := []row{
		{"H3-01", "Cha mẹ có đăng ký kết hôn.", "da_ket_hon", true, "có đăng ký kết hôn", true},
		{"H3-02", "Cha mẹ chưa đăng ký kết hôn.", "da_ket_hon", false, "chưa đăng ký kết hôn", true},
		{"H3-03", "Cha mẹ không có đăng ký kết hôn.", "da_ket_hon", true, "có đăng ký kết hôn", false},
		{"H3-04", "Cha mẹ không có đăng ký kết hôn.", "da_ket_hon", false, "không có đăng ký kết hôn", true},
		{"H3-05", "Mẹ đi đăng ký.", "nguoi_di_dang_ky", "me", "mẹ đi đăng ký", true},
		{"H3-06", "Cha đi đăng ký.", "nguoi_di_dang_ky", "cha", "cha đi đăng ký", true},
		{"H3-07", "Không phải mẹ đi đăng ký, mà cha đi đăng ký.", "nguoi_di_dang_ky", "me", "mẹ đi đăng ký", false},
		{"H3-08", "Không phải mẹ đi đăng ký, mà cha đi đăng ký.", "nguoi_di_dang_ky", "cha", "cha đi đăng ký", true},
		{"H3-09", "Tôi không có giấy chứng sinh.", "co_giay_chung_sinh", true, "có giấy chứng sinh", false},
		{"H3-10", "Tôi không có giấy chứng sinh.", "co_giay_chung_sinh", false, "không có giấy chứng sinh", true},
		{"H3-11", "Tôi có giấy chứng sinh.", "co_giay_chung_sinh", true, "có giấy chứng sinh", true},
		{"H3-12", "Để tôi kiểm tra lại.", "da_ket_hon", true, "kiểm tra lại", false},
		{"H3-13", "Tôi nói nhầm, cha mẹ chưa đăng ký kết hôn.", "da_ket_hon", false, "chưa đăng ký kết hôn", true},
		{"H3-14", "Mẹ đi đăng ký, không phải cha.", "nguoi_di_dang_ky", "cha", "cha", false},
		{"H3-punct", "Cha mẹ   không có   đăng ký kết hôn.", "da_ket_hon", true, "có đăng ký kết hôn", false},
		{"H3-nfc", "Cha mẹ không có đăng ký kết hôn.", "da_ket_hon", false, "không có đăng ký kết hôn", true},
		{"H3-other-clause", "Tôi có giấy chứng sinh, chưa kết hôn.", "co_giay_chung_sinh", true, "có giấy chứng sinh", true},
		{"H3-other-clause-marriage-true", "Tôi có giấy chứng sinh, chưa kết hôn.", "da_ket_hon", true, "kết hôn", false},
		{"H3-contrast-me-ok", "Mẹ đi đăng ký, không phải cha.", "nguoi_di_dang_ky", "me", "mẹ đi đăng ký", true},
		{"H3-invented", "Tôi muốn đăng ký khai sinh.", "da_ket_hon", true, "đã đăng ký kết hôn", false},
		{"H3-empty-ev", "Cha mẹ có đăng ký kết hôn.", "da_ket_hon", true, "", false},
		{"H3-contradict", "Cha mẹ có đăng ký kết hôn, cha mẹ chưa đăng ký kết hôn.", "da_ket_hon", true, "có đăng ký kết hôn", false},
		{"H3-chang", "Cha mẹ chẳng có đăng ký kết hôn.", "da_ket_hon", true, "có đăng ký kết hôn", false},
		{"H3-khong-phai-married", "Cha mẹ không phải đã đăng ký kết hôn.", "da_ket_hon", true, "đã đăng ký kết hôn", false},
	}
	for _, tc := range cases {
		t.Run(tc.id, func(t *testing.T) {
			ok, reason := GroundSlot(tc.message, spec(tc.key), tc.key, tc.value, tc.evidence)
			if ok != tc.accept {
				t.Fatalf("accept=%v want %v reason=%s msg=%q key=%s val=%v", ok, tc.accept, reason, tc.message, tc.key, tc.value)
			}
		})
	}
}

func TestH3OriginalCounterexamplesRejected(t *testing.T) {
	ok, reason := GroundSlot(
		"Cha mẹ không có đăng ký kết hôn.",
		spec("da_ket_hon"),
		"da_ket_hon",
		true,
		"có đăng ký kết hôn",
	)
	if ok {
		t.Fatalf("H3-A must reject da_ket_hon=true, reason=%s", reason)
	}
	ok, reason = GroundSlot(
		"Không phải mẹ đi đăng ký, mà cha đi đăng ký.",
		spec("nguoi_di_dang_ky"),
		"nguoi_di_dang_ky",
		"me",
		"mẹ đi đăng ký",
	)
	if ok {
		t.Fatalf("H3-B must reject nguoi_di_dang_ky=me, reason=%s", reason)
	}
}

func TestH3ValidateDoesNotFlipWrongBoolean(t *testing.T) {
	msg := "Cha mẹ không có đăng ký kết hôn."
	resp := groundedResp([]SlotResult{
		{Key: "da_ket_hon", Value: true, Confidence: 1.0, Evidence: "có đăng ký kết hôn", Operation: "set"},
	})
	out, err := Validate(resp, khaiSinhCandidates(), "dk_khai_sinh", []string{"da_ket_hon"}, 0.6, nil, msg)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := out.Slots["da_ket_hon"]; ok {
		t.Fatalf("must drop wrong true, not convert it, got %#v", out.Slots["da_ket_hon"])
	}
}

func TestH3CorrectionStillAccepted(t *testing.T) {
	msg := "Tôi nói nhầm, cha mẹ chưa đăng ký kết hôn."
	resp := groundedResp([]SlotResult{
		{Key: "da_ket_hon", Value: false, Confidence: 0.95, Evidence: "chưa đăng ký kết hôn", Operation: "correct"},
	})
	current := map[string]CurrentSlot{"da_ket_hon": {Filled: true}}
	out, err := Validate(resp, khaiSinhCandidates(), "dk_khai_sinh", []string{"da_ket_hon"}, 0.6, current, msg)
	if err != nil {
		t.Fatal(err)
	}
	if v, ok := out.Slots["da_ket_hon"]; !ok || v.Value != false || v.Operation != "correct" {
		t.Fatalf("want correct false, got %#v", out.Slots)
	}
}
