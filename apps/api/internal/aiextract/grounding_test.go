package aiextract

import "testing"

func khaiSinhCandidates() []Candidate {
	return []Candidate{{
		ProcedureCode: "dk_khai_sinh",
		Name:          "Đăng ký khai sinh",
		Slots: map[string]SlotSpec{
			"noi_sinh":           {Type: "string"},
			"da_ket_hon":         {Type: "boolean"},
			"co_giay_chung_sinh": {Type: "boolean"},
			"nguoi_di_dang_ky":   {Type: "enum", EnumValues: []string{"cha", "me", "ong_ba", "nguoi_duoc_uy_quyen"}},
		},
	}}
}

func groundedResp(slots []SlotResult) *Response {
	code := "dk_khai_sinh"
	return stamp(&Response{
		SchemaVersion:         SchemaVersion,
		Intent:                &IntentResult{ProcedureCode: &code, Confidence: 0.95},
		SlotsForProcedureCode: &code,
		Slots:                 slots,
	})
}

func TestExampleA_MissingMarriageRemainsUnknown(t *testing.T) {
	msg := "Tôi muốn đăng ký khai sinh cho con, đã có giấy chứng sinh, mẹ đi đăng ký."
	resp := groundedResp([]SlotResult{
		{Key: "co_giay_chung_sinh", Value: true, Confidence: 0.99, Evidence: "đã có giấy chứng sinh", Operation: "set"},
		{Key: "nguoi_di_dang_ky", Value: "me", Confidence: 0.99, Evidence: "mẹ đi đăng ký", Operation: "set"},
		{Key: "da_ket_hon", Value: true, Confidence: 1.0, Evidence: "mẹ đi đăng ký", Operation: "set"},
	})
	out, err := Validate(resp, khaiSinhCandidates(), "", nil, 0.6, nil, msg)
	if err != nil {
		t.Fatalf("Validate: %v", err)
	}
	if _, ok := out.Slots["da_ket_hon"]; ok {
		t.Fatalf("da_ket_hon must stay unknown, got %#v", out.Slots["da_ket_hon"])
	}
	if v := out.Slots["co_giay_chung_sinh"]; v.Value != true {
		t.Fatalf("co_giay_chung_sinh=%#v", v)
	}
	if v := out.Slots["nguoi_di_dang_ky"]; v.Value != "me" {
		t.Fatalf("nguoi_di_dang_ky=%#v", v)
	}
}

func TestExampleB_ExplicitUnmarried(t *testing.T) {
	msg := "Cha mẹ chưa đăng ký kết hôn."
	resp := groundedResp([]SlotResult{
		{Key: "da_ket_hon", Value: false, Confidence: 0.95, Evidence: "Cha mẹ chưa đăng ký kết hôn", Operation: "set"},
	})
	out, err := Validate(resp, khaiSinhCandidates(), "dk_khai_sinh", []string{"da_ket_hon"}, 0.6, nil, msg)
	if err != nil {
		t.Fatal(err)
	}
	if v, ok := out.Slots["da_ket_hon"]; !ok || v.Value != false {
		t.Fatalf("want da_ket_hon=false, got %#v", out.Slots)
	}
}

func TestExampleC_ExplicitMarried(t *testing.T) {
	msg := "Cha mẹ đã đăng ký kết hôn."
	resp := groundedResp([]SlotResult{
		{Key: "da_ket_hon", Value: true, Confidence: 0.95, Evidence: "đã đăng ký kết hôn", Operation: "set"},
	})
	out, err := Validate(resp, khaiSinhCandidates(), "dk_khai_sinh", []string{"da_ket_hon"}, 0.6, nil, msg)
	if err != nil {
		t.Fatal(err)
	}
	if v, ok := out.Slots["da_ket_hon"]; !ok || v.Value != true {
		t.Fatalf("want da_ket_hon=true, got %#v", out.Slots)
	}
}

func TestExampleD_AmbiguousDoesNotChange(t *testing.T) {
	msg := "Để tôi kiểm tra lại giấy tờ."
	resp := groundedResp([]SlotResult{
		{Key: "da_ket_hon", Value: true, Confidence: 0.9, Evidence: "giấy tờ", Operation: "set"},
		{Key: "co_giay_chung_sinh", Value: true, Confidence: 0.9, Evidence: "giấy tờ", Operation: "set"},
	})
	out, err := Validate(resp, khaiSinhCandidates(), "dk_khai_sinh", []string{"da_ket_hon", "co_giay_chung_sinh"}, 0.6, nil, msg)
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Slots) != 0 {
		t.Fatalf("ambiguous message must not persist slots, got %#v", out.Slots)
	}
}

func TestExampleE_CorrectionUpdatesMarriage(t *testing.T) {
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
		t.Fatalf("want correct da_ket_hon=false, got %#v", out.Slots)
	}
}

func TestExampleF_MotherBringingDocumentDoesNotInferMarriage(t *testing.T) {
	msg := "Mẹ mang giấy chứng sinh lên xã."
	resp := groundedResp([]SlotResult{
		{Key: "da_ket_hon", Value: true, Confidence: 1.0, Evidence: "Mẹ mang giấy chứng sinh", Operation: "set"},
		{Key: "co_giay_chung_sinh", Value: true, Confidence: 0.9, Evidence: "giấy chứng sinh", Operation: "set"},
	})
	out, err := Validate(resp, khaiSinhCandidates(), "", nil, 0.6, nil, msg)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := out.Slots["da_ket_hon"]; ok {
		t.Fatal("must not infer marriage from bringing a birth certificate")
	}
}

func TestExampleG_NegationNotFlippedToPositive(t *testing.T) {
	msg := "Tôi không có giấy chứng sinh."
	resp := groundedResp([]SlotResult{
		{Key: "co_giay_chung_sinh", Value: true, Confidence: 0.99, Evidence: "có giấy chứng sinh", Operation: "set"},
	})
	out, err := Validate(resp, khaiSinhCandidates(), "", nil, 0.6, nil, msg)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := out.Slots["co_giay_chung_sinh"]; ok {
		t.Fatalf("positive extraction must not survive negation, got %#v", out.Slots)
	}
	respFalse := groundedResp([]SlotResult{
		{Key: "co_giay_chung_sinh", Value: false, Confidence: 0.99, Evidence: "không có giấy chứng sinh", Operation: "set"},
	})
	out2, err := Validate(respFalse, khaiSinhCandidates(), "", nil, 0.6, nil, msg)
	if err != nil {
		t.Fatal(err)
	}
	if v := out2.Slots["co_giay_chung_sinh"]; v.Value != false {
		t.Fatalf("want co_giay_chung_sinh=false, got %#v", out2.Slots)
	}
}

func TestInventedEvidenceSpanIsDropped(t *testing.T) {
	msg := "Tôi muốn đăng ký khai sinh cho con."
	resp := groundedResp([]SlotResult{
		{Key: "da_ket_hon", Value: true, Confidence: 1.0, Evidence: "đã đăng ký kết hôn", Operation: "set"},
	})
	out, err := Validate(resp, khaiSinhCandidates(), "", nil, 0.6, nil, msg)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := out.Slots["da_ket_hon"]; ok {
		t.Fatal("invented evidence span must be dropped")
	}
}

func TestE2E01Shape_HospitalBirthCertificateMother(t *testing.T) {
	msg := "Tôi muốn đăng ký khai sinh cho con sinh tại Bệnh viện huyện Chư Sê, đã có giấy chứng sinh, mẹ đi đăng ký."
	resp := groundedResp([]SlotResult{
		{Key: "noi_sinh", Value: "Bệnh viện huyện Chư Sê", Confidence: 0.95, Evidence: "Bệnh viện huyện Chư Sê", Operation: "set"},
		{Key: "co_giay_chung_sinh", Value: true, Confidence: 0.95, Evidence: "đã có giấy chứng sinh", Operation: "set"},
		{Key: "nguoi_di_dang_ky", Value: "me", Confidence: 0.95, Evidence: "mẹ đi đăng ký", Operation: "set"},
		{Key: "da_ket_hon", Value: true, Confidence: 1.0, Evidence: "mẹ đi đăng ký", Operation: "set"},
	})
	out, err := Validate(resp, khaiSinhCandidates(), "", nil, 0.6, nil, msg)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := out.Slots["da_ket_hon"]; ok {
		t.Fatal("H1: da_ket_hon must not be guessed from mother registering")
	}
	if out.Slots["noi_sinh"].Value != "Bệnh viện huyện Chư Sê" {
		t.Fatalf("noi_sinh=%#v", out.Slots["noi_sinh"])
	}
	if out.Slots["co_giay_chung_sinh"].Value != true {
		t.Fatalf("co_giay_chung_sinh=%#v", out.Slots["co_giay_chung_sinh"])
	}
	if out.Slots["nguoi_di_dang_ky"].Value != "me" {
		t.Fatalf("nguoi_di_dang_ky=%#v", out.Slots["nguoi_di_dang_ky"])
	}
}

func TestEmptyEvidenceDropsSlot(t *testing.T) {
	msg := "Cha mẹ đã đăng ký kết hôn."
	resp := groundedResp([]SlotResult{
		{Key: "da_ket_hon", Value: true, Confidence: 0.99, Evidence: "", Operation: "set"},
	})
	out, err := Validate(resp, khaiSinhCandidates(), "", nil, 0.6, nil, msg)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := out.Slots["da_ket_hon"]; ok {
		t.Fatal("empty evidence must drop the slot")
	}
}
