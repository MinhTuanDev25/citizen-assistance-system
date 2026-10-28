package decision

import (
	"strings"
	"testing"
)

// The live khai_sinh_v1 shape: two enum questions, one with labels.
func khaiSinhEnums(labels bool) Definition {
	lbl := ""
	if labels {
		lbl = `, "enum_labels": ["Trẻ sinh bình thường", "Trẻ bị bỏ rơi", "Mang thai hộ", "Chưa xác định được cha, mẹ"]`
	}
	def, err := ParseDefinition([]byte(`{
		"procedure_code": "khai_sinh_v1", "name": "Đăng ký khai sinh",
		"slots": {
			"hinh_thuc_nop": {"type": "enum", "question": "Bạn muốn nộp hồ sơ theo hình thức nào?",
				"enum_values": ["truc_tiep", "truc_tuyen", "buu_chinh"], "enum_labels": ["Trực tiếp", "Trực tuyến", "Qua bưu chính"]},
			"truong_hop_tre": {"type": "enum", "question": "Trường hợp khai sinh của trẻ thuộc diện nào sau đây?",
				"enum_values": ["tre_sinh_thuong", "tre_bi_bo_roi", "mang_thai_ho", "chua_xac_dinh_cha_me"]` + lbl + `},
			"co_giay_chung_sinh": {"type": "boolean", "question": "Bạn có Giấy chứng sinh của trẻ không?"}
		},
		"required_slots": ["hinh_thuc_nop", "truong_hop_tre"],
		"conditional_slots": [{"when": {"slot": "truong_hop_tre", "equals": "tre_sinh_thuong"}, "require": ["co_giay_chung_sinh"]}],
		"guidance": {"summary": "s", "checklist": ["Tờ khai", "Giấy chứng sinh"], "where_to_submit": "UBND xã"}
	}`))
	if err != nil {
		panic(err)
	}
	return def
}

func TestAskListsEnumChoices(t *testing.T) {
	out := Decide(TurnInput{Definition: khaiSinhEnums(true), Message: "tôi muốn đăng kí giấy khai sinh cho con"})
	if out.Action != ActionAsk {
		t.Fatalf("action = %s", out.Action)
	}
	for _, want := range []string{
		"1) Bạn muốn nộp hồ sơ theo hình thức nào?\n   a) Trực tiếp\n   b) Trực tuyến\n   c) Qua bưu chính",
		"2) Trường hợp khai sinh của trẻ thuộc diện nào sau đây?\n   a) Trẻ sinh bình thường\n   b) Trẻ bị bỏ rơi\n   c) Mang thai hộ\n   d) Chưa xác định được cha, mẹ",
	} {
		if !strings.Contains(out.ReplyText, want) {
			t.Fatalf("reply must list the choices:\n%s", out.ReplyText)
		}
	}
	if len(out.Questions) != 2 || len(out.Questions[1].Options) != 4 || out.Questions[1].Options[1] != (QuestionOption{Value: "tre_bi_bo_roi", Label: "Trẻ bị bỏ rơi"}) {
		t.Fatalf("questions = %+v", out.Questions)
	}
}

func TestAskWithoutLabelsFallsBackToCodes(t *testing.T) {
	out := Decide(TurnInput{Definition: khaiSinhEnums(false), Message: "khai sinh"})
	if !strings.Contains(out.ReplyText, "   a) tre sinh thuong\n   b) tre bi bo roi") {
		t.Fatalf("reply = %s", out.ReplyText)
	}
}

// The conversation from the screenshot, continued: "Trực tiếp", then a bare
// choice, then the conditional question, then the checklist.
func TestChoiceAnswersReachChecklist(t *testing.T) {
	for _, answer := range []string{"a", "A)", "1", "chọn a", "phương án 1", "Trẻ sinh bình thường", "trẻ sinh bình thường ạ"} {
		def := khaiSinhEnums(true)
		first := Decide(TurnInput{Definition: def, Message: "Trực tiếp"})
		if first.SlotState["hinh_thuc_nop"].Value != "truc_tiep" {
			t.Fatalf("hinh_thuc_nop = %+v", first.SlotState["hinh_thuc_nop"])
		}
		second := Decide(TurnInput{Definition: def, Prior: first.SlotState, HadPriorState: true, Message: answer})
		if second.SlotState["truong_hop_tre"].Value != "tre_sinh_thuong" {
			t.Fatalf("%q: truong_hop_tre = %+v (reply %s)", answer, second.SlotState["truong_hop_tre"], second.ReplyText)
		}
		if second.Action != ActionAsk || len(second.Questions) != 1 || second.Questions[0].Slot != "co_giay_chung_sinh" {
			t.Fatalf("%q: the conditional question must follow: %+v", answer, second.Questions)
		}
		third := Decide(TurnInput{Definition: def, Prior: second.SlotState, HadPriorState: true, Message: "có"})
		if third.Action != ActionFinal || !strings.Contains(third.ReplyText, "Hồ sơ cần chuẩn bị:\n1) Tờ khai\n2) Giấy chứng sinh") {
			t.Fatalf("%q: checklist expected, got %s: %s", answer, third.Action, third.ReplyText)
		}
	}
}

func TestBareChoiceIsIgnoredWhenAmbiguousOrOutOfRange(t *testing.T) {
	def := khaiSinhEnums(true)
	// Two enum questions open: "b" could answer either.
	start := Decide(TurnInput{Definition: def, Message: "khai sinh"})
	out := Decide(TurnInput{Definition: def, Prior: start.SlotState, HadPriorState: true, Message: "b"})
	if out.SlotState["hinh_thuc_nop"].Filled() || out.SlotState["truong_hop_tre"].Filled() {
		t.Fatalf("an ambiguous bare choice must not fill a slot: %+v", out.SlotState)
	}
	first := Decide(TurnInput{Definition: def, Message: "Trực tiếp"})
	for _, bad := range []string{"e", "5", "0"} {
		out := Decide(TurnInput{Definition: def, Prior: first.SlotState, HadPriorState: true, Message: bad})
		if out.SlotState["truong_hop_tre"].Filled() {
			t.Fatalf("%q is not a choice of 4 options: %+v", bad, out.SlotState["truong_hop_tre"])
		}
	}
}

func TestEnumLabelsMismatchIsADraftError(t *testing.T) {
	raw := []byte(`{"procedure_code": "khai_sinh_v1", "name": "n",
		"slots": {"k": {"type": "enum", "question": "?", "enum_values": ["a", "b"], "enum_labels": ["A"]}},
		"required_slots": ["k"], "guidance": {"summary": "s", "checklist": ["x"], "where_to_submit": "w"}}`)
	found := false
	for _, is := range ValidateDraftDefinition(raw, "khai_sinh_v1").Issues {
		found = found || (is.Code == "ENUM_LABELS_MISMATCH" && is.Severity == "error")
	}
	if !found {
		t.Fatal("labels that do not match the values one to one must be an error")
	}
}

// The 09:12 conversation: after the checklist the citizen says "à quên trẻ bị
// bỏ rơi". The change is said back, and the birth-certificate answer (asked
// only for tre_sinh_thuong) no longer stays in the state.
func TestCorrectionIsAnnouncedAndStaleConditionalDropped(t *testing.T) {
	def := khaiSinhEnums(true)
	prior := map[string]SlotValue{
		"hinh_thuc_nop":      {Value: "truc_tiep", Status: StatusKnown},
		"truong_hop_tre":     {Value: "tre_sinh_thuong", Status: StatusKnown},
		"co_giay_chung_sinh": {Value: true, Status: StatusKnown},
	}
	out, applied := DecideWithExtraction(TurnInput{Definition: def, Prior: prior, HadPriorState: true, Message: "à quên trẻ bị bỏ rơi"},
		&ExtractionProposal{Slots: map[string]ExtractedSlot{"truong_hop_tre": {Value: "tre_bi_bo_roi", Operation: "correct"}}})
	if len(applied.Accepted) != 1 || out.SlotState["truong_hop_tre"].Value != "tre_bi_bo_roi" {
		t.Fatalf("state = %+v applied = %+v", out.SlotState, applied)
	}
	if !strings.HasPrefix(out.ReplyText, "Mình cập nhật:\n- Trường hợp khai sinh của trẻ thuộc diện nào sau đây? — Trẻ bị bỏ rơi.") {
		t.Fatalf("the correction must be said back:\n%s", out.ReplyText)
	}
	if _, kept := out.SlotState["co_giay_chung_sinh"]; kept {
		t.Fatalf("the birth-certificate answer no longer applies: %+v", out.SlotState)
	}
	if out.Action != ActionFinal {
		t.Fatalf("action = %s", out.Action)
	}

	// Switching back asks the birth-certificate question again.
	back, _ := DecideWithExtraction(TurnInput{Definition: def, Prior: out.SlotState, HadPriorState: true, Message: "nhầm, trẻ sinh bình thường"},
		&ExtractionProposal{Slots: map[string]ExtractedSlot{"truong_hop_tre": {Value: "tre_sinh_thuong", Operation: "correct"}}})
	if back.Action != ActionAsk || len(back.Questions) != 1 || back.Questions[0].Slot != "co_giay_chung_sinh" {
		t.Fatalf("switching back must ask again: %s %+v", back.Action, back.Questions)
	}
}
