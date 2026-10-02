package aiextract

import (
	"math"
	"testing"
)

func strPtr(s string) *string { return &s }

func stamp(r *Response) *Response {
	if r == nil {
		return nil
	}
	if r.Provider == "" {
		r.Provider = "mock"
	}
	if r.Model == "" {
		r.Model = "mock-1"
	}
	return r
}

func baseCandidates() []Candidate {
	return []Candidate{
		{
			ProcedureCode: "dk_khai_sinh",
			Name:          "Đăng ký khai sinh",
			Slots: map[string]SlotSpec{
				"nguoi_di_dang_ky": {Type: "enum", EnumValues: []string{"cha", "me"}},
				"da_ket_hon":       {Type: "boolean"},
				"ho_ten_be":        {Type: "string"},
				"so_luong":         {Type: "number"},
			},
		},
		{ProcedureCode: "dk_bhyt_ho_gia_dinh", Name: "BHYT hộ gia đình", Slots: map[string]SlotSpec{}},
	}
}

func TestValidateAcceptsWellFormedIntentAndSlot(t *testing.T) {
	resp := &Response{
		SchemaVersion:         SchemaVersion,
		Intent:                &IntentResult{ProcedureCode: strPtr("dk_khai_sinh"), Confidence: 0.9},
		SlotsForProcedureCode: strPtr("dk_khai_sinh"),
		Slots: []SlotResult{
			{Key: "nguoi_di_dang_ky", Value: "cha", Confidence: 0.9, Operation: "set"},
		},
	}
	out, err := Validate(stamp(resp), baseCandidates(), "", []string{}, 0.5, nil)
	if err != nil {
		t.Fatalf("Validate: %v", err)
	}
	if out.IntentProcedureCode != "dk_khai_sinh" {
		t.Fatalf("IntentProcedureCode = %q", out.IntentProcedureCode)
	}
	if v, ok := out.Slots["nguoi_di_dang_ky"]; !ok || v.Value != "cha" {
		t.Fatalf("slot not accepted: %+v", out.Slots)
	}
}

func TestValidateRejectsUnknownProcedureCode(t *testing.T) {
	resp := &Response{
		SchemaVersion: SchemaVersion,
		Intent:        &IntentResult{ProcedureCode: strPtr("does_not_exist"), Confidence: 0.9},
	}
	if _, err := Validate(stamp(resp), baseCandidates(), "", nil, 0.5, nil); err == nil {
		t.Fatal("expected error for unknown procedure_code")
	}
}

func TestValidateRejectsUnknownSlotKey(t *testing.T) {
	resp := &Response{
		SchemaVersion:         SchemaVersion,
		SlotsForProcedureCode: strPtr("dk_khai_sinh"),
		Slots:                 []SlotResult{{Key: "not_a_real_slot", Value: "x", Confidence: 0.9, Operation: "set"}},
	}
	if _, err := Validate(stamp(resp), baseCandidates(), "dk_khai_sinh", []string{"not_a_real_slot"}, 0.5, nil); err == nil {
		t.Fatal("expected error for unknown slot key")
	}
}

func TestValidateEnforcesAllowedKeysOnlyForPinnedProcedure(t *testing.T) {
	resp := &Response{
		SchemaVersion:         SchemaVersion,
		SlotsForProcedureCode: strPtr("dk_khai_sinh"),
		Slots:                 []SlotResult{{Key: "ho_ten_be", Value: "An", Confidence: 0.9, Operation: "set"}},
	}
	// Pinned to dk_khai_sinh but allowed_slot_keys does NOT include ho_ten_be.
	if _, err := Validate(stamp(resp), baseCandidates(), "dk_khai_sinh", []string{"da_ket_hon"}, 0.5, nil); err == nil {
		t.Fatal("expected error: slot outside allowed_slot_keys while pinned")
	}
	// Same slot is fine when this procedure was *just selected* (not pinned
	// yet) — slots_for_procedure_code matches the freshly selected intent,
	// so allowed_slot_keys (a restriction that only applies to the
	// already-pinned procedure) is not enforced.
	freshSelect := &Response{
		SchemaVersion:         SchemaVersion,
		Intent:                &IntentResult{ProcedureCode: strPtr("dk_khai_sinh"), Confidence: 0.9},
		SlotsForProcedureCode: strPtr("dk_khai_sinh"),
		Slots:                 resp.Slots,
	}
	if _, err := Validate(stamp(freshSelect), baseCandidates(), "", []string{"da_ket_hon"}, 0.5, nil); err != nil {
		t.Fatalf("fresh-selection slot should be accepted regardless of allowed_slot_keys: %v", err)
	}
}

func TestValidateRejectsBooleanTypeMismatch(t *testing.T) {
	resp := &Response{
		SchemaVersion:         SchemaVersion,
		SlotsForProcedureCode: strPtr("dk_khai_sinh"),
		Slots:                 []SlotResult{{Key: "da_ket_hon", Value: "yes", Confidence: 0.9, Operation: "set"}},
	}
	if _, err := Validate(stamp(resp), baseCandidates(), "dk_khai_sinh", []string{"da_ket_hon"}, 0.5, nil); err == nil {
		t.Fatal("expected type mismatch error for boolean slot given a string")
	}
}

func TestValidateRejectsEnumValueOutsideAllowlist(t *testing.T) {
	resp := &Response{
		SchemaVersion:         SchemaVersion,
		SlotsForProcedureCode: strPtr("dk_khai_sinh"),
		Slots:                 []SlotResult{{Key: "nguoi_di_dang_ky", Value: "ba_con", Confidence: 0.9, Operation: "set"}},
	}
	if _, err := Validate(stamp(resp), baseCandidates(), "dk_khai_sinh", []string{"nguoi_di_dang_ky"}, 0.5, nil); err == nil {
		t.Fatal("expected error for enum value not in enum_values")
	}
}

func TestValidateRejectsNonFiniteNumber(t *testing.T) {
	resp := &Response{
		SchemaVersion:         SchemaVersion,
		SlotsForProcedureCode: strPtr("dk_khai_sinh"),
		Slots:                 []SlotResult{{Key: "so_luong", Value: math.Inf(1), Confidence: 0.9, Operation: "set"}},
	}
	if _, err := Validate(stamp(resp), baseCandidates(), "dk_khai_sinh", []string{"so_luong"}, 0.5, nil); err == nil {
		t.Fatal("expected error for non-finite number")
	}
}

func TestValidateRejectsCorrectWithoutExistingValue(t *testing.T) {
	resp := &Response{
		SchemaVersion:         SchemaVersion,
		SlotsForProcedureCode: strPtr("dk_khai_sinh"),
		Slots:                 []SlotResult{{Key: "da_ket_hon", Value: true, Confidence: 0.9, Operation: "correct"}},
	}
	current := map[string]CurrentSlot{"da_ket_hon": {Filled: false}}
	if _, err := Validate(stamp(resp), baseCandidates(), "dk_khai_sinh", []string{"da_ket_hon"}, 0.5, current); err == nil {
		t.Fatal("expected error: correct against an unfilled slot")
	}
	current["da_ket_hon"] = CurrentSlot{Filled: true}
	if _, err := Validate(stamp(resp), baseCandidates(), "dk_khai_sinh", []string{"da_ket_hon"}, 0.5, current); err != nil {
		t.Fatalf("correct against a filled slot should be accepted: %v", err)
	}
}

func TestValidateRejectsInvalidOperation(t *testing.T) {
	resp := &Response{
		SchemaVersion:         SchemaVersion,
		SlotsForProcedureCode: strPtr("dk_khai_sinh"),
		Slots:                 []SlotResult{{Key: "da_ket_hon", Value: true, Confidence: 0.9, Operation: "delete"}},
	}
	if _, err := Validate(stamp(resp), baseCandidates(), "dk_khai_sinh", []string{"da_ket_hon"}, 0.5, nil); err == nil {
		t.Fatal("expected error for invalid operation")
	}
}

func TestValidateRejectsWrongSchemaVersion(t *testing.T) {
	resp := &Response{SchemaVersion: "extract.v2"}
	if _, err := Validate(stamp(resp), baseCandidates(), "", nil, 0.5, nil); err == nil {
		t.Fatal("expected error for wrong schema_version")
	}
}

func TestValidateRejectsSlotsForUnrelatedProcedure(t *testing.T) {
	resp := &Response{
		SchemaVersion:         SchemaVersion,
		Intent:                &IntentResult{ProcedureCode: strPtr("dk_khai_sinh"), Confidence: 0.9},
		SlotsForProcedureCode: strPtr("dk_bhyt_ho_gia_dinh"), // neither selected intent nor pinned
		Slots:                 []SlotResult{{Key: "x", Value: "y", Confidence: 0.9, Operation: "set"}},
	}
	if _, err := Validate(stamp(resp), baseCandidates(), "", nil, 0.5, nil); err == nil {
		t.Fatal("expected error: slots target neither selected intent nor pinned procedure")
	}
}

func TestValidateDropsSlotBelowConfidenceThresholdWithoutError(t *testing.T) {
	resp := &Response{
		SchemaVersion:         SchemaVersion,
		SlotsForProcedureCode: strPtr("dk_khai_sinh"),
		Slots:                 []SlotResult{{Key: "da_ket_hon", Value: true, Confidence: 0.1, Operation: "set"}},
	}
	out, err := Validate(stamp(resp), baseCandidates(), "dk_khai_sinh", []string{"da_ket_hon"}, 0.6, nil)
	if err != nil {
		t.Fatalf("low-confidence slot should be dropped, not rejected: %v", err)
	}
	if _, ok := out.Slots["da_ket_hon"]; ok {
		t.Fatal("low-confidence slot should not be present in output")
	}
}

func TestValidateRejectsDuplicateSlotKeys(t *testing.T) {
	resp := &Response{
		SchemaVersion:         SchemaVersion,
		SlotsForProcedureCode: strPtr("dk_khai_sinh"),
		Slots: []SlotResult{
			{Key: "da_ket_hon", Value: true, Confidence: 0.9, Operation: "set"},
			{Key: "da_ket_hon", Value: false, Confidence: 0.9, Operation: "set"},
		},
	}
	if _, err := Validate(stamp(resp), baseCandidates(), "dk_khai_sinh", []string{"da_ket_hon"}, 0.5, nil); err == nil {
		t.Fatal("expected error for duplicate slot keys")
	}
}

func TestValidateRejectsOutOfRangeConfidence(t *testing.T) {
	resp := &Response{
		SchemaVersion: SchemaVersion,
		Intent:        &IntentResult{ProcedureCode: strPtr("dk_khai_sinh"), Confidence: 1.5},
	}
	if _, err := Validate(stamp(resp), baseCandidates(), "", nil, 0.5, nil); err == nil {
		t.Fatal("expected error for out-of-range intent confidence")
	}
}

func TestValidateNilResponseRejected(t *testing.T) {
	if _, err := Validate(nil, baseCandidates(), "", nil, 0.5, nil); err == nil {
		t.Fatal("expected error for nil response")
	}
}

func TestValidateRejectsNullIntentWithNonZeroConfidence(t *testing.T) {
	resp := stamp(&Response{
		SchemaVersion: SchemaVersion,
		Intent:        &IntentResult{ProcedureCode: nil, Confidence: 0.4},
	})
	if _, err := Validate(resp, baseCandidates(), "", nil, 0.5, nil); err == nil {
		t.Fatal("expected error for null intent with non-zero confidence")
	}
}

func TestValidateRejectsNullIntentWithBadAlternative(t *testing.T) {
	resp := stamp(&Response{
		SchemaVersion: SchemaVersion,
		Intent: &IntentResult{
			ProcedureCode: nil,
			Confidence:    0,
			Alternatives:  []Alternative{{ProcedureCode: "not_real", Confidence: 0.4}},
		},
	})
	if _, err := Validate(resp, baseCandidates(), "", nil, 0.5, nil); err == nil {
		t.Fatal("expected error for invalid alternative on a null intent")
	}
}

func TestValidateRejectsDuplicateAlternatives(t *testing.T) {
	code := "dk_khai_sinh"
	resp := stamp(&Response{
		SchemaVersion: SchemaVersion,
		Intent: &IntentResult{
			ProcedureCode: &code,
			Confidence:    0.9,
			Alternatives: []Alternative{
				{ProcedureCode: "dk_bhyt_ho_gia_dinh", Confidence: 0.4},
				{ProcedureCode: "dk_bhyt_ho_gia_dinh", Confidence: 0.3},
			},
		},
	})
	if _, err := Validate(resp, baseCandidates(), "", nil, 0.5, nil); err == nil {
		t.Fatal("expected error for duplicate alternatives")
	}
}

func TestValidateRejectsSpoofedProvider(t *testing.T) {
	resp := &Response{
		SchemaVersion: SchemaVersion,
		Provider:      "PII-TOKEN-7788",
		Model:         "mock-1",
	}
	if _, err := Validate(resp, baseCandidates(), "", nil, 0.5, nil); err == nil {
		t.Fatal("expected error for provider outside the allowlist")
	}
}

func TestValidateAcceptsNullIntentAtZeroConfidence(t *testing.T) {
	resp := stamp(&Response{
		SchemaVersion: SchemaVersion,
		Intent:        &IntentResult{ProcedureCode: nil, Confidence: 0},
	})
	out, err := Validate(resp, baseCandidates(), "", nil, 0.5, nil)
	if err != nil {
		t.Fatal(err)
	}
	if out.IntentProcedureCode != "" || len(out.Alternatives) != 0 {
		t.Fatalf("null intent must not select anything: %+v", out)
	}
	if out.Provider != "mock" || out.Model != "mock-1" {
		t.Fatalf("observability = %s %s", out.Provider, out.Model)
	}
}
