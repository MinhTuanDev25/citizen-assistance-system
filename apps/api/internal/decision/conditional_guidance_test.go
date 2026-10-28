package decision

import (
	"strings"
	"testing"
)

func khaiSinhConditional() Definition {
	def, err := ParseDefinition([]byte(`{
		"procedure_code": "khai_sinh_v1", "name": "Đăng ký khai sinh",
		"slots": {
			"hinh_thuc_nop": {"type": "enum", "question": "Hình thức nộp?", "enum_values": ["truc_tiep", "truc_tuyen"], "enum_labels": ["Trực tiếp", "Trực tuyến"]},
			"truong_hop_tre": {"type": "enum", "question": "Trường hợp của trẻ?", "enum_values": ["tre_sinh_thuong", "tre_bi_bo_roi"], "enum_labels": ["Trẻ sinh bình thường", "Trẻ bị bỏ rơi"]},
			"co_giay_chung_sinh": {"type": "boolean", "question": "Có Giấy chứng sinh không?"}
		},
		"required_slots": ["hinh_thuc_nop", "truong_hop_tre"],
		"conditional_slots": [{"when": {"slot": "truong_hop_tre", "equals": "tre_sinh_thuong"}, "require": ["co_giay_chung_sinh"]}],
		"guidance": {
			"summary": "Đăng ký khai sinh tại UBND xã.",
			"checklist": ["Giấy tờ tùy thân của người đi đăng ký"],
			"conditional_checklist": [
				{"when": {"slot": "hinh_thuc_nop", "equals": "truc_tiep"}, "item": "Tờ khai đăng ký khai sinh (bản giấy)"},
				{"when": {"slot": "hinh_thuc_nop", "equals": "truc_tuyen"}, "item": "Mẫu hộ tịch điện tử tương tác"},
				{"when": {"slot": "truong_hop_tre", "equals": "tre_sinh_thuong"}, "item": "Giấy chứng sinh"},
				{"when": {"slot": "truong_hop_tre", "equals": "tre_bi_bo_roi"}, "item": "Biên bản về việc trẻ bị bỏ rơi"},
				{"when": {"slot": "co_giay_chung_sinh", "equals": false}, "item": "Văn bản của người làm chứng về việc sinh"}
			],
			"notes": ["Đăng ký trong 60 ngày kể từ ngày sinh."],
			"conditional_notes": [
				{"when": {"slot": "truong_hop_tre", "equals": "tre_bi_bo_roi"}, "item": "UBND xã niêm yết 07 ngày trước khi đăng ký."},
				{"when": {"slot": "hinh_thuc_nop", "equals": "truc_tuyen"}, "item": "Kiểm tra và xác nhận Giấy khai sinh điện tử trong 01 ngày."}
			]
		}
	}`))
	if err != nil {
		panic(err)
	}
	return def
}

func finalReply(t *testing.T, def Definition, state map[string]SlotValue) TurnOutput {
	t.Helper()
	out := Decide(TurnInput{Definition: def, Prior: state, HadPriorState: true, Message: "ok"})
	if out.Action != ActionFinal {
		t.Fatalf("action = %s (%s)", out.Action, out.ReplyText)
	}
	return out
}

// The screenshots: "trực tiếp" + "trẻ sinh thường" must not get the online
// form or the abandoned-child papers.
func TestGuidanceShowsOnlyWhatMatchesTheAnswers(t *testing.T) {
	def := khaiSinhConditional()
	out := finalReply(t, def, map[string]SlotValue{
		"hinh_thuc_nop":      {Value: "truc_tiep", Status: StatusKnown},
		"truong_hop_tre":     {Value: "tre_sinh_thuong", Status: StatusKnown},
		"co_giay_chung_sinh": {Value: true, Status: StatusKnown},
	})
	want := "Hồ sơ cần chuẩn bị:\n1) Giấy tờ tùy thân của người đi đăng ký\n2) Tờ khai đăng ký khai sinh (bản giấy)\n3) Giấy chứng sinh\n"
	if !strings.Contains(out.ReplyText, want) {
		t.Fatalf("checklist:\n%s", out.ReplyText)
	}
	for _, unwanted := range []string{"Mẫu hộ tịch điện tử", "bị bỏ rơi", "niêm yết", "người làm chứng", "điện tử trong 01 ngày"} {
		if strings.Contains(out.ReplyText, unwanted) {
			t.Fatalf("%q does not apply to this citizen:\n%s", unwanted, out.ReplyText)
		}
	}
	if len(out.Guidance.Checklist) != 3 || out.Guidance.ConditionalChecklist != nil {
		t.Fatalf("guidance sent to the client must be resolved: %+v", out.Guidance)
	}
}

func TestGuidanceForAbandonedChildOnline(t *testing.T) {
	out := finalReply(t, khaiSinhConditional(), map[string]SlotValue{
		"hinh_thuc_nop":  {Value: "truc_tuyen", Status: StatusKnown},
		"truong_hop_tre": {Value: "tre_bi_bo_roi", Status: StatusKnown},
	})
	for _, want := range []string{"Mẫu hộ tịch điện tử tương tác", "Biên bản về việc trẻ bị bỏ rơi", "niêm yết 07 ngày", "Giấy khai sinh điện tử trong 01 ngày", "Đăng ký trong 60 ngày"} {
		if !strings.Contains(out.ReplyText, want) {
			t.Fatalf("missing %q:\n%s", want, out.ReplyText)
		}
	}
	for _, unwanted := range []string{"bản giấy", "Giấy chứng sinh\n"} {
		if strings.Contains(out.ReplyText, unwanted) {
			t.Fatalf("%q does not apply:\n%s", unwanted, out.ReplyText)
		}
	}
}

func TestBooleanConditionAndNoCertificate(t *testing.T) {
	out := finalReply(t, khaiSinhConditional(), map[string]SlotValue{
		"hinh_thuc_nop":      {Value: "truc_tiep", Status: StatusKnown},
		"truong_hop_tre":     {Value: "tre_sinh_thuong", Status: StatusKnown},
		"co_giay_chung_sinh": {Value: false, Status: StatusKnown},
	})
	if !strings.Contains(out.ReplyText, "Văn bản của người làm chứng về việc sinh") {
		t.Fatalf("a false answer must select its items:\n%s", out.ReplyText)
	}
}

func TestInvalidGuidanceConditionIsADraftError(t *testing.T) {
	cases := map[string]string{
		"unknown slot":    `{"when": {"slot": "khong_co", "equals": "x"}, "item": "A"}`,
		"unknown answer":  `{"when": {"slot": "truong_hop_tre", "equals": "tre_mo_coi"}, "item": "A"}`,
		"bool for enum":   `{"when": {"slot": "truong_hop_tre", "equals": true}, "item": "A"}`,
		"string for bool": `{"when": {"slot": "co", "equals": "co"}, "item": "A"}`,
		"string question": `{"when": {"slot": "noi_sinh", "equals": "x"}, "item": "A"}`,
		"empty item":      `{"when": {"slot": "co", "equals": true}, "item": " "}`,
	}
	for name, item := range cases {
		raw := []byte(`{"procedure_code": "khai_sinh_v1", "name": "n",
			"slots": {
				"truong_hop_tre": {"type": "enum", "question": "?", "enum_values": ["tre_sinh_thuong", "tre_bi_bo_roi"]},
				"co": {"type": "boolean", "question": "?"},
				"noi_sinh": {"type": "string", "question": "?"}
			},
			"required_slots": ["truong_hop_tre"],
			"guidance": {"summary": "s", "checklist": ["x"], "where_to_submit": "w", "conditional_checklist": [` + item + `]}}`)
		found := false
		for _, is := range ValidateDraftDefinition(raw, "khai_sinh_v1").Issues {
			found = found || (is.Code == "INVALID_GUIDANCE_CONDITION" && is.ProposalID == "conditional_checklist:0")
		}
		if !found {
			t.Errorf("%s: expected INVALID_GUIDANCE_CONDITION", name)
		}
	}
}

// The 8.56970.897 case: co_giay_chung_sinh is neither required nor required
// by a rule, and co_nguoi_lam_chung depends on it, so neither is ever asked.
func TestUnreachableQuestionIsADraftError(t *testing.T) {
	raw := []byte(`{"procedure_code": "khai_sinh_v1", "name": "n",
		"slots": {
			"truong_hop_tre": {"type": "enum", "question": "?", "enum_values": ["tre_sinh_thuong", "tre_bi_bo_roi"]},
			"co_giay_chung_sinh": {"type": "boolean", "question": "?"},
			"co_nguoi_lam_chung": {"type": "boolean", "question": "?"}
		},
		"required_slots": ["truong_hop_tre"],
		"conditional_slots": [{"when": {"slot": "co_giay_chung_sinh", "equals": false}, "require": ["co_nguoi_lam_chung"]}],
		"guidance": {"summary": "s", "checklist": ["x"], "where_to_submit": "w"}}`)
	unreachable := map[string]bool{}
	for _, is := range ValidateDraftDefinition(raw, "khai_sinh_v1").Issues {
		if is.Code == "UNREACHABLE_SLOT" && is.Severity == "error" {
			unreachable[is.ProposalID] = true
		}
	}
	if !unreachable["slot:co_giay_chung_sinh"] || unreachable["slot:co_nguoi_lam_chung"] || unreachable["slot:truong_hop_tre"] {
		t.Fatalf("unreachable = %v", unreachable)
	}
	// Adding the missing rule fixes it.
	fixed := []byte(strings.Replace(string(raw), `"conditional_slots": [`,
		`"conditional_slots": [{"when": {"slot": "truong_hop_tre", "equals": "tre_sinh_thuong"}, "require": ["co_giay_chung_sinh"]}, `, 1))
	for _, is := range ValidateDraftDefinition(fixed, "khai_sinh_v1").Issues {
		if is.Code == "UNREACHABLE_SLOT" {
			t.Fatalf("still unreachable: %+v", is)
		}
	}
}
