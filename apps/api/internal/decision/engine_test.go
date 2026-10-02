package decision

import (
	"encoding/json"
	"strings"
	"testing"
)

func khaiSinh() Definition {
	raw := []byte(`{
		"procedure_code": "dk_khai_sinh",
		"name": "Đăng ký khai sinh",
		"intent_examples": [
			"Tôi muốn làm giấy khai sinh cho con",
			"Đăng ký khai sinh",
			"Làm khai sinh cho bé mới sinh"
		],
		"slots": {
			"noi_sinh": {"type": "string", "question": "Bé sinh ở đâu?"},
			"da_ket_hon": {"type": "boolean", "question": "Cha mẹ bé đã đăng ký kết hôn chưa?"},
			"co_giay_chung_sinh": {"type": "boolean", "question": "Anh/chị có giấy chứng sinh không?"},
			"nguoi_di_dang_ky": {
				"type": "enum",
				"enum_values": ["cha", "me", "ong_ba", "nguoi_duoc_uy_quyen"],
				"question": "Ai sẽ đi đăng ký khai sinh?"
			}
		},
		"required_slots": ["noi_sinh", "da_ket_hon", "co_giay_chung_sinh"],
		"conditional_slots": [{
			"when": {"slot": "co_giay_chung_sinh", "equals": false},
			"require": ["nguoi_di_dang_ky"]
		}],
		"guidance": {
			"summary": "Chuẩn bị hồ sơ đăng ký khai sinh.",
			"checklist": ["Giấy chứng sinh", "CCCD cha mẹ"],
			"where_to_submit": "UBND xã",
			"notes": ["Nên đăng ký sớm."]
		},
		"citations": [{"doc_id": "seed_dk_khai_sinh_v1", "title": "Hướng dẫn đăng ký khai sinh", "source_type": "manual_seed"}]
	}`)
	def, err := ParseDefinition(raw)
	if err != nil {
		panic(err)
	}
	return def
}

func chungThuc() Definition {
	raw := []byte(`{
		"procedure_code": "chung_thuc_ban_sao",
		"name": "Chứng thực bản sao",
		"intent_examples": ["Chứng thực bản sao CCCD", "Tôi muốn chứng thực bằng đại học"],
		"slots": {
			"loai_giay_to": {"type": "string", "question": "Loại giấy tờ nào?"}
		},
		"required_slots": [],
		"guidance": {
			"summary": "Mang bản chính và bản photo đến UBND xã.",
			"checklist": ["Bản chính", "Bản photo", "CCCD"],
			"where_to_submit": "Bộ phận một cửa",
			"notes": ["Seed tạm."]
		},
		"citations": [{"doc_id": "seed_chung_thuc", "title": "Hướng dẫn chứng thực", "source_type": "manual_seed"}]
	}`)
	def, err := ParseDefinition(raw)
	if err != nil {
		panic(err)
	}
	return def
}

func TestDecideKhaiSinhAsksAllMissing(t *testing.T) {
	out := Decide(TurnInput{
		Definition: khaiSinh(),
		Message:    "Tôi muốn làm giấy khai sinh cho con tôi.",
	})
	if out.Action != ActionAsk {
		t.Fatalf("action = %s", out.Action)
	}
	want := []string{"noi_sinh", "da_ket_hon", "co_giay_chung_sinh"}
	if strings.Join(out.MissingSlots, ",") != strings.Join(want, ",") {
		t.Fatalf("missing = %#v", out.MissingSlots)
	}
	if len(out.Questions) != 3 {
		t.Fatalf("questions = %#v", out.Questions)
	}
	if out.SlotState["noi_sinh"].Status != StatusMissing {
		t.Fatalf("intent sentence filled noi_sinh: %#v", out.SlotState["noi_sinh"])
	}
	if out.ReplyText == "" || !strings.Contains(out.ReplyText, "1)") || !strings.Contains(out.ReplyText, "3)") {
		t.Fatalf("reply = %q", out.ReplyText)
	}
}

func TestDecideNumberedAnswersThenFinal(t *testing.T) {
	def := khaiSinh()
	first := Decide(TurnInput{Definition: def, Message: "Đăng ký khai sinh"})
	second := Decide(TurnInput{
		Definition:    def,
		Prior:         first.SlotState,
		HadPriorState: true,
		Message:       "1. Bệnh viện Đa khoa tỉnh\n2. rồi\n3. có",
	})
	if second.Action != ActionFinal {
		t.Fatalf("action = %s missing=%#v reply=%s", second.Action, second.MissingSlots, second.ReplyText)
	}
	if second.FilledSlots["noi_sinh"] != "Bệnh viện Đa khoa tỉnh" {
		t.Fatalf("noi_sinh = %#v", second.FilledSlots["noi_sinh"])
	}
	if second.FilledSlots["da_ket_hon"] != true || second.FilledSlots["co_giay_chung_sinh"] != true {
		t.Fatalf("filled = %#v", second.FilledSlots)
	}
	if second.Guidance == nil || len(second.Citations) != 1 {
		t.Fatalf("guidance/citations missing")
	}
}

func TestDecideConditionalSlot(t *testing.T) {
	def := khaiSinh()
	first := Decide(TurnInput{Definition: def, Message: "khai sinh"})
	second := Decide(TurnInput{
		Definition:    def,
		Prior:         first.SlotState,
		HadPriorState: true,
		Message:       "1. sinh tại nhà\n2. chưa\n3. không",
	})
	if second.Action != ActionAsk {
		t.Fatalf("action = %s missing=%#v", second.Action, second.MissingSlots)
	}
	if strings.Join(second.MissingSlots, ",") != "nguoi_di_dang_ky" {
		t.Fatalf("missing = %#v", second.MissingSlots)
	}
	third := Decide(TurnInput{
		Definition:    def,
		Prior:         second.SlotState,
		HadPriorState: true,
		Message:       "ông bà",
	})
	if third.Action != ActionFinal {
		t.Fatalf("action = %s missing=%#v state=%#v", third.Action, third.MissingSlots, third.SlotState["nguoi_di_dang_ky"])
	}
	if third.FilledSlots["nguoi_di_dang_ky"] != "ong_ba" {
		t.Fatalf("enum = %#v", third.FilledSlots["nguoi_di_dang_ky"])
	}
}

func TestDecideClauseAnswerFillsStringLeftover(t *testing.T) {
	def := khaiSinh()
	first := Decide(TurnInput{Definition: def, Message: "Đăng ký khai sinh"})
	second := Decide(TurnInput{
		Definition:    def,
		Prior:         first.SlotState,
		HadPriorState: true,
		Message:       "Bệnh viện đa khoa, đã kết hôn, có giấy chứng sinh",
	})
	if second.Action != ActionFinal {
		t.Fatalf("action = %s missing=%#v state=%#v", second.Action, second.MissingSlots, second.SlotState)
	}
}

func TestDecideDirectWhenNoRequiredSlots(t *testing.T) {
	out := Decide(TurnInput{
		Definition: chungThuc(),
		Message:    "Chứng thực bản sao CCCD",
	})
	if out.Action != ActionDirect {
		t.Fatalf("action = %s", out.Action)
	}
	if len(out.Citations) != 1 || out.Guidance == nil {
		t.Fatalf("expected guidance + citation")
	}
}

func TestMatchIntent(t *testing.T) {
	candidates := []IntentCandidate{
		{ProcedureCode: "dk_khai_sinh", Name: "Đăng ký khai sinh", Examples: khaiSinh().IntentExamples},
		{ProcedureCode: "chung_thuc_ban_sao", Name: "Chứng thực bản sao", Examples: append(chungThuc().IntentExamples, "Công chứng photo giấy tờ tại xã")},
		{ProcedureCode: "dk_bhyt_ho_gia_dinh", Name: "Đăng ký BHYT hộ gia đình", Examples: []string{"Đăng ký BHYT hộ gia đình", "Mua bảo hiểm y tế cho cả nhà"}},
	}
	got, ok := MatchIntent("Tôi muốn làm giấy khai sinh cho con tôi.", candidates)
	if !ok || got.ProcedureCode != "dk_khai_sinh" {
		t.Fatalf("got %#v ok=%v", got, ok)
	}
	got, ok = MatchIntent("Chứng thực bản sao", candidates)
	if !ok || got.ProcedureCode != "chung_thuc_ban_sao" {
		t.Fatalf("chung thuc got %#v ok=%v", got, ok)
	}
	if _, ok := MatchIntent("đăng ký", candidates); ok {
		t.Fatal("ambiguous dang ky should not match")
	}
	if _, ok := MatchIntent("xin chào buổi sáng", candidates); ok {
		t.Fatal("greeting should not match")
	}
	got, ok = MatchIntent("Tôi muốn công chứng", candidates)
	if !ok || got.ProcedureCode != "chung_thuc_ban_sao" {
		t.Fatalf("cong chung got %#v ok=%v", got, ok)
	}
	got, ok = MatchIntent("Tôi muốn đăng ký giấy khai sinh", candidates)
	if !ok || got.ProcedureCode != "dk_khai_sinh" {
		t.Fatalf("khai sinh got %#v ok=%v", got, ok)
	}
}

func TestDecidePlaceAnswerFillsOnlyBirthplace(t *testing.T) {
	def := khaiSinh()
	first := Decide(TurnInput{Definition: def, Message: "Đăng ký khai sinh"})
	second := Decide(TurnInput{
		Definition:    def,
		Prior:         first.SlotState,
		HadPriorState: true,
		Message:       "ở bệnh viện hùng vogw",
	})
	if second.SlotState["noi_sinh"].Value != "ở bệnh viện hùng vogw" {
		t.Fatalf("noi_sinh = %#v", second.SlotState["noi_sinh"])
	}
	if second.Action != ActionAsk || len(second.MissingSlots) != 2 {
		t.Fatalf("action=%s missing=%#v", second.Action, second.MissingSlots)
	}
}

func TestDecideCommaAnswersInQuestionOrder(t *testing.T) {
	def := khaiSinh()
	first := Decide(TurnInput{Definition: def, Message: "Đăng ký khai sinh"})
	second := Decide(TurnInput{
		Definition:    def,
		Prior:         first.SlotState,
		HadPriorState: true,
		Message:       "ở bệnh vien, chưa, ko",
	})
	if second.SlotState["noi_sinh"].Value != "ở bệnh vien" {
		t.Fatalf("noi_sinh = %#v", second.SlotState["noi_sinh"])
	}
	if second.SlotState["da_ket_hon"].Value != false || second.SlotState["co_giay_chung_sinh"].Value != false {
		t.Fatalf("bools = %#v %#v", second.SlotState["da_ket_hon"], second.SlotState["co_giay_chung_sinh"])
	}
	if strings.Join(second.MissingSlots, ",") != "nguoi_di_dang_ky" {
		t.Fatalf("missing = %#v", second.MissingSlots)
	}
}

func TestDecideCorrectsFilledBoolean(t *testing.T) {
	def := khaiSinh()
	prior := map[string]SlotValue{
		"da_ket_hon": {Value: true, Status: StatusKnown},
	}
	out := Decide(TurnInput{
		Definition:    def,
		Prior:         prior,
		HadPriorState: true,
		Message:       "à quên chưa kết hôn",
	})
	if out.SlotState["da_ket_hon"].Value != false {
		t.Fatalf("da_ket_hon = %#v", out.SlotState["da_ket_hon"])
	}
	if out.SlotState["noi_sinh"].Filled() {
		t.Fatalf("correction became birthplace: %#v", out.SlotState["noi_sinh"])
	}
	if !strings.Contains(out.ReplyText, "Mình cập nhật") {
		t.Fatalf("reply = %q", out.ReplyText)
	}
}

func TestDecideCorrectsAfterGuidance(t *testing.T) {
	def := khaiSinh()
	prior := map[string]SlotValue{
		"noi_sinh":           {Value: "bv hùng vương", Status: StatusKnown},
		"da_ket_hon":         {Value: false, Status: StatusKnown},
		"co_giay_chung_sinh": {Value: true, Status: StatusKnown},
	}
	if !HasSlotCorrection(def, prior, "à quên đã kết hôn rồi") {
		t.Fatal("expected a correction")
	}
	out := Decide(TurnInput{
		Definition:    def,
		Prior:         prior,
		HadPriorState: true,
		Message:       "à quên đã kết hôn rồi",
	})
	if out.Action != ActionFinal {
		t.Fatalf("action = %s missing=%#v reply=%s", out.Action, out.MissingSlots, out.ReplyText)
	}
	if out.SlotState["da_ket_hon"].Value != true {
		t.Fatalf("da_ket_hon = %#v", out.SlotState["da_ket_hon"])
	}
	if !strings.Contains(out.ReplyText, "Mình cập nhật") {
		t.Fatalf("reply = %q", out.ReplyText)
	}
}

func TestDecideRejectsNumberedNonBoolean(t *testing.T) {
	def := khaiSinh()
	first := Decide(TurnInput{Definition: def, Message: "Đăng ký khai sinh"})
	second := Decide(TurnInput{
		Definition:    def,
		Prior:         first.SlotState,
		HadPriorState: true,
		Message:       "1. Gia Lai\n2. Haha",
	})
	if second.SlotState["noi_sinh"].Value != "Gia Lai" {
		t.Fatalf("noi_sinh = %#v", second.SlotState["noi_sinh"])
	}
	if second.SlotState["da_ket_hon"].Filled() {
		t.Fatalf("haha filled marriage: %#v", second.SlotState["da_ket_hon"])
	}
	if !strings.Contains(second.ReplyText, "Mình chưa hiểu") {
		t.Fatalf("reply = %q", second.ReplyText)
	}
}

func TestEqualsFalse(t *testing.T) {
	def := khaiSinh()
	state := map[string]SlotValue{
		"co_giay_chung_sinh": {Value: false, Status: StatusKnown},
	}
	missing := def.Missing(state)
	if !containsStr(missing, "nguoi_di_dang_ky") {
		t.Fatalf("missing = %#v", missing)
	}
	raw := json.RawMessage("false")
	if !equalsValue(false, raw) {
		t.Fatal("false should match")
	}
}
