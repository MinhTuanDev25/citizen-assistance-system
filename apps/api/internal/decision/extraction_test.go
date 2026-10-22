package decision

import "testing"

func khaiSinhDef() Definition {
	def, err := ParseDefinition([]byte(`{
		"procedure_code": "dk_khai_sinh",
		"name": "Đăng ký khai sinh",
		"intent_examples": ["đăng ký khai sinh"],
		"slots": {
			"nguoi_di_dang_ky": {"type": "enum", "question": "Ai đi đăng ký?", "enum_values": ["cha", "me"]},
			"da_ket_hon": {"type": "boolean", "question": "Cha mẹ đã đăng ký kết hôn chưa?"},
			"ho_ten_be": {"type": "string", "question": "Họ tên bé?"}
		},
		"required_slots": ["nguoi_di_dang_ky", "da_ket_hon", "ho_ten_be"],
		"guidance": {"summary": "s", "checklist": ["a"], "where_to_submit": "UBND xã"},
		"citations": []
	}`))
	if err != nil {
		panic(err)
	}
	return def
}

func TestDecideWithExtractionNilProposalMatchesPlainDecide(t *testing.T) {
	def := khaiSinhDef()
	in := TurnInput{Definition: def, Prior: map[string]SlotValue{}, Message: "xin chào"}
	want := Decide(in)
	got, applied := DecideWithExtraction(in, nil)
	if got.Action != want.Action || got.ReplyText != want.ReplyText {
		t.Fatalf("nil extraction changed output: got=%+v want=%+v", got, want)
	}
	if len(applied.Accepted) != 0 || len(applied.Rejected) != 0 {
		t.Fatalf("expected no applied slots, got %+v", applied)
	}
}

func TestDecideWithExtractionFillsMissingSlot(t *testing.T) {
	def := khaiSinhDef()
	in := TurnInput{Definition: def, Prior: map[string]SlotValue{}, Message: "ba"}
	extraction := &ExtractionProposal{Slots: map[string]ExtractedSlot{
		"nguoi_di_dang_ky": {Value: "cha", Operation: "set"},
	}}
	out, applied := DecideWithExtraction(in, extraction)
	if out.SlotState["nguoi_di_dang_ky"].Value != "cha" {
		t.Fatalf("slot not seeded: %+v", out.SlotState)
	}
	if len(applied.Accepted) != 1 || applied.Accepted[0] != "nguoi_di_dang_ky" {
		t.Fatalf("expected accepted=[nguoi_di_dang_ky], got %+v", applied)
	}
	// Still missing the other two required slots — action must still ask.
	if out.Action != ActionAsk {
		t.Fatalf("Action = %s, want %s (AI partial fill must not skip missing slots)", out.Action, ActionAsk)
	}
}

func TestDecideWithExtractionNeverOverwritesFilledSlotOnSet(t *testing.T) {
	def := khaiSinhDef()
	prior := map[string]SlotValue{"nguoi_di_dang_ky": {Value: "me", Status: StatusKnown}}
	in := TurnInput{Definition: def, Prior: prior, HadPriorState: true, Message: "ba"}
	extraction := &ExtractionProposal{Slots: map[string]ExtractedSlot{
		"nguoi_di_dang_ky": {Value: "cha", Operation: "set"},
	}}
	out, applied := DecideWithExtraction(in, extraction)
	if out.SlotState["nguoi_di_dang_ky"].Value != "me" {
		t.Fatalf("set must not overwrite an already-filled slot, got %v", out.SlotState["nguoi_di_dang_ky"].Value)
	}
	if len(applied.Rejected) != 1 || applied.Rejected[0] != "nguoi_di_dang_ky" {
		t.Fatalf("expected rejected=[nguoi_di_dang_ky], got %+v", applied)
	}
}

func TestDecideWithExtractionCorrectOverwritesFilledSlot(t *testing.T) {
	def := khaiSinhDef()
	prior := map[string]SlotValue{"nguoi_di_dang_ky": {Value: "me", Status: StatusKnown}}
	in := TurnInput{Definition: def, Prior: prior, HadPriorState: true, Message: "à quên là ba"}
	extraction := &ExtractionProposal{Slots: map[string]ExtractedSlot{
		"nguoi_di_dang_ky": {Value: "cha", Operation: "correct"},
	}}
	out, applied := DecideWithExtraction(in, extraction)
	if out.SlotState["nguoi_di_dang_ky"].Value != "cha" {
		t.Fatalf("correct should overwrite, got %v", out.SlotState["nguoi_di_dang_ky"].Value)
	}
	if len(applied.Accepted) != 1 {
		t.Fatalf("expected 1 accepted correction, got %+v", applied)
	}
}

func TestDecideWithExtractionRejectsUnknownSlotKey(t *testing.T) {
	def := khaiSinhDef()
	in := TurnInput{Definition: def, Prior: map[string]SlotValue{}, Message: "x"}
	extraction := &ExtractionProposal{Slots: map[string]ExtractedSlot{
		"not_a_real_slot": {Value: "x", Operation: "set"},
	}}
	out, applied := DecideWithExtraction(in, extraction)
	if _, ok := out.SlotState["not_a_real_slot"]; ok {
		t.Fatal("unknown slot key must never appear in slot state")
	}
	if len(applied.Rejected) != 1 || applied.Rejected[0] != "not_a_real_slot" {
		t.Fatalf("expected rejected=[not_a_real_slot], got %+v", applied)
	}
}

func TestDecideWithExtractionNeverTouchesGuidanceOrCitations(t *testing.T) {
	def := khaiSinhDef()
	prior := map[string]SlotValue{
		"nguoi_di_dang_ky": {Value: "cha", Status: StatusKnown},
		"da_ket_hon":       {Value: true, Status: StatusKnown},
	}
	in := TurnInput{Definition: def, Prior: prior, HadPriorState: true, Message: "tên bé là An"}
	extraction := &ExtractionProposal{Slots: map[string]ExtractedSlot{
		"ho_ten_be": {Value: "An", Operation: "set"},
	}}
	withAI, _ := DecideWithExtraction(in, extraction)
	plain := Decide(TurnInput{
		Definition: def,
		Prior: map[string]SlotValue{
			"nguoi_di_dang_ky": {Value: "cha", Status: StatusKnown},
			"da_ket_hon":       {Value: true, Status: StatusKnown},
			"ho_ten_be":        {Value: "An", Status: StatusKnown},
		},
		HadPriorState: true,
		Message:       "tên bé là An",
	})
	if withAI.Guidance == nil || plain.Guidance == nil || withAI.Guidance.Summary != plain.Guidance.Summary {
		t.Fatalf("guidance diverged between AI-seeded and plain decide")
	}
	if len(withAI.Citations) != len(plain.Citations) {
		t.Fatalf("citations diverged")
	}
}

func TestPolicyBandMatchesDefaultThresholds(t *testing.T) {
	p := DefaultPolicy
	if p.Band(0.9) != IntentSelect {
		t.Fatalf("0.9 should be select band")
	}
	if p.Band(0.6) != IntentConfirm {
		t.Fatalf("0.6 should be confirm band")
	}
	if p.Band(0.1) != IntentNone {
		t.Fatalf("0.1 should be none band")
	}
	// Exact boundary values.
	if p.Band(p.SelectMin) != IntentSelect {
		t.Fatalf("SelectMin boundary should be select")
	}
	if p.Band(p.ConfirmMin) != IntentConfirm {
		t.Fatalf("ConfirmMin boundary should be confirm")
	}
}

func TestDecideWithExtractionKeepsMarriageUnknownWhenNotSeeded(t *testing.T) {
	def := khaiSinh()
	msg := "Tôi muốn đăng ký khai sinh cho con sinh tại Bệnh viện huyện Chư Sê, đã có giấy chứng sinh, mẹ đi đăng ký."
	extraction := &ExtractionProposal{Slots: map[string]ExtractedSlot{
		"noi_sinh":           {Value: "Bệnh viện huyện Chư Sê", Operation: "set"},
		"co_giay_chung_sinh": {Value: true, Operation: "set"},
		"nguoi_di_dang_ky":   {Value: "me", Operation: "set"},
	}}
	out, applied := DecideWithExtraction(TurnInput{Definition: def, Message: msg}, extraction)
	if out.SlotState["da_ket_hon"].Filled() {
		t.Fatalf("da_ket_hon must remain unknown, got %#v", out.SlotState["da_ket_hon"])
	}
	if out.SlotState["noi_sinh"].Value != "Bệnh viện huyện Chư Sê" {
		t.Fatalf("noi_sinh=%#v", out.SlotState["noi_sinh"])
	}
	if out.SlotState["co_giay_chung_sinh"].Value != true {
		t.Fatalf("co_giay_chung_sinh=%#v", out.SlotState["co_giay_chung_sinh"])
	}
	if out.SlotState["nguoi_di_dang_ky"].Value != "me" {
		t.Fatalf("nguoi_di_dang_ky=%#v", out.SlotState["nguoi_di_dang_ky"])
	}
	if len(applied.Accepted) != 3 {
		t.Fatalf("accepted=%v", applied.Accepted)
	}
	if out.Action != ActionAsk {
		t.Fatalf("action=%s missing=%v (want ASK for da_ket_hon)", out.Action, out.MissingSlots)
	}
	if !containsStr(out.MissingSlots, "da_ket_hon") {
		t.Fatalf("missing=%v want da_ket_hon", out.MissingSlots)
	}
}

func TestKeywordDoesNotFillUncertainMarriage(t *testing.T) {
	def := khaiSinh()
	prior := map[string]SlotValue{
		"noi_sinh":           {Value: "Bệnh viện huyện Chư Sê", Status: StatusKnown},
		"co_giay_chung_sinh": {Value: true, Status: StatusKnown},
		"nguoi_di_dang_ky":   {Value: "me", Status: StatusKnown},
	}
	out := Decide(TurnInput{
		Definition:    def,
		Prior:         prior,
		HadPriorState: true,
		Message:       "Cha mẹ không biết đã đăng ký kết hôn chưa.",
	})
	if out.SlotState["da_ket_hon"].Filled() {
		t.Fatalf("uncertain message must not fill da_ket_hon, got %#v", out.SlotState["da_ket_hon"])
	}
	if out.Action != ActionAsk {
		t.Fatalf("action=%s want ASK_MISSING_SLOTS", out.Action)
	}
}

func TestDecideExplicitUnmarriedFillsOnlyMarriage(t *testing.T) {
	def := khaiSinh()
	prior := map[string]SlotValue{
		"noi_sinh":           {Value: "Bệnh viện huyện Chư Sê", Status: StatusKnown},
		"co_giay_chung_sinh": {Value: true, Status: StatusKnown},
	}
	out := Decide(TurnInput{Definition: def, Prior: prior, HadPriorState: true, Message: "Cha mẹ chưa đăng ký kết hôn."})
	if out.SlotState["da_ket_hon"].Value != false {
		t.Fatalf("da_ket_hon=%#v", out.SlotState["da_ket_hon"])
	}
	if out.SlotState["co_giay_chung_sinh"].Value != true {
		t.Fatalf("document slot changed: %#v", out.SlotState["co_giay_chung_sinh"])
	}
}

func TestCustomPolicyOverridesThresholds(t *testing.T) {
	p := Policy{SelectMin: 0.7, ConfirmMin: 0.4}
	if p.Band(0.75) != IntentSelect {
		t.Fatalf("0.75 should be select under custom policy")
	}
	if p.Band(0.5) != IntentConfirm {
		t.Fatalf("0.5 should be confirm under custom policy")
	}
	if p.Band(0.3) != IntentNone {
		t.Fatalf("0.3 should be none under custom policy")
	}
}
