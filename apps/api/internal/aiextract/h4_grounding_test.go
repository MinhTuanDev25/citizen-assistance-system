package aiextract

import "testing"

func TestH4GroundingMatrix(t *testing.T) {
	type row struct {
		id, message, key string
		value            any
		evidence         string
		accept           bool
	}
	msgUnsureMarry := "Cha mẹ không biết đã đăng ký kết hôn chưa."
	msgUnsureWho := "Không rõ mẹ đi đăng ký hay cha đi đăng ký."
	msgNoWant := "Tôi không muốn mẹ đi đăng ký."
	msgMixed := "Tôi không rõ ai đi đăng ký, nhưng tôi có giấy chứng sinh."
	msgContrast := "Không phải mẹ đi đăng ký, mà cha đi đăng ký."
	cases := []row{
		{"H4-01", msgUnsureMarry, "da_ket_hon", true, "đã đăng ký kết hôn", false},
		{"H4-02", msgUnsureMarry, "da_ket_hon", false, "đã đăng ký kết hôn chưa", false},
		{"H4-03", msgUnsureWho, "nguoi_di_dang_ky", "me", "mẹ đi đăng ký", false},
		{"H4-04", msgUnsureWho, "nguoi_di_dang_ky", "cha", "cha đi đăng ký", false},
		{"H4-05", msgNoWant, "nguoi_di_dang_ky", "me", "mẹ đi đăng ký", false},
		{"H4-06", msgNoWant, "nguoi_di_dang_ky", "cha", "cha", false},
		{"H4-07", "Cha mẹ đã đăng ký kết hôn.", "da_ket_hon", true, "đã đăng ký kết hôn", true},
		{"H4-08", "Cha mẹ chưa đăng ký kết hôn.", "da_ket_hon", false, "chưa đăng ký kết hôn", true},
		{"H4-09", msgMixed, "co_giay_chung_sinh", true, "có giấy chứng sinh", true},
		{"H4-10", msgMixed, "nguoi_di_dang_ky", "me", "đi đăng ký", false},
		{"H4-11", "Tôi nói nhầm, cha mẹ chưa đăng ký kết hôn.", "da_ket_hon", false, "chưa đăng ký kết hôn", true},
		{"H4-12", "Hình như cha mẹ đã đăng ký kết hôn.", "da_ket_hon", true, "đã đăng ký kết hôn", false},
		{"H4-13", "Cha mẹ đã đăng ký kết hôn nhưng tôi không chắc.", "da_ket_hon", true, "đã đăng ký kết hôn", false},
		{"H4-14", msgContrast, "nguoi_di_dang_ky", "cha", "cha đi đăng ký", true},
		{"H4-15", msgContrast, "nguoi_di_dang_ky", "me", "mẹ đi đăng ký", false},
		{"H4-16", "Tôi không có giấy chứng sinh.", "co_giay_chung_sinh", false, "không có giấy chứng sinh", true},
		{"H4-punct", "Cha mẹ  không  biết đã đăng ký kết hôn chưa.", "da_ket_hon", true, "đã đăng ký kết hôn", false},
		{"H4-case", "CHA MẸ KHÔNG BIẾT ĐÃ ĐĂNG KÝ KẾT HÔN CHƯA.", "da_ket_hon", true, "ĐÃ ĐĂNG KÝ KẾT HÔN", false},
		{"H4-maybe", "Có lẽ cha mẹ đã đăng ký kết hôn.", "da_ket_hon", true, "đã đăng ký kết hôn", false},
		{"H4-invented", msgUnsureMarry, "da_ket_hon", true, "đã cưới từ năm ngoái", false},
		{"H4-empty-ev", msgUnsureMarry, "da_ket_hon", true, "", false},
	}
	for _, tc := range cases {
		t.Run(tc.id, func(t *testing.T) {
			ok, reason := GroundSlot(tc.message, spec(tc.key), tc.key, tc.value, tc.evidence)
			if ok != tc.accept {
				t.Fatalf("accept=%v want %v reason=%s key=%s val=%v", ok, tc.accept, reason, tc.key, tc.value)
			}
		})
	}
}

func TestH4OriginalCounterexamplesRejected(t *testing.T) {
	ok, reason := GroundSlot("Cha mẹ không biết đã đăng ký kết hôn chưa.", spec("da_ket_hon"), "da_ket_hon", true, "đã đăng ký kết hôn")
	if ok {
		t.Fatalf("H4-A must reject true, reason=%s", reason)
	}
	ok, reason = GroundSlot("Không rõ mẹ đi đăng ký hay cha đi đăng ký.", spec("nguoi_di_dang_ky"), "nguoi_di_dang_ky", "me", "mẹ đi đăng ký")
	if ok {
		t.Fatalf("H4-B must reject me, reason=%s", reason)
	}
	ok, reason = GroundSlot("Tôi không muốn mẹ đi đăng ký.", spec("nguoi_di_dang_ky"), "nguoi_di_dang_ky", "me", "mẹ đi đăng ký")
	if ok {
		t.Fatalf("H4-C must reject me, reason=%s", reason)
	}
}

func TestH4DoesNotConvertPreferenceToOpposite(t *testing.T) {
	ok, _ := GroundSlot("Tôi không muốn mẹ đi đăng ký.", spec("nguoi_di_dang_ky"), "nguoi_di_dang_ky", "cha", "mẹ đi đăng ký")
	if ok {
		t.Fatal("must not convert không muốn mẹ into cha")
	}
}

func TestH4EmptyMessageDrops(t *testing.T) {
	ok, reason := GroundSlot("", spec("da_ket_hon"), "da_ket_hon", true, "đã đăng ký kết hôn")
	if ok {
		t.Fatalf("empty message must drop, reason=%s", reason)
	}
}

func TestH4ValidateDropsUncertainWithoutFlipping(t *testing.T) {
	msg := "Cha mẹ không biết đã đăng ký kết hôn chưa."
	resp := groundedResp([]SlotResult{
		{Key: "da_ket_hon", Value: true, Confidence: 1.0, Evidence: "đã đăng ký kết hôn", Operation: "set"},
	})
	out, err := Validate(resp, khaiSinhCandidates(), "dk_khai_sinh", []string{"da_ket_hon"}, 0.6, nil, msg)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := out.Slots["da_ket_hon"]; ok {
		t.Fatalf("uncertain true must be dropped not flipped, got %#v", out.Slots["da_ket_hon"])
	}
}
