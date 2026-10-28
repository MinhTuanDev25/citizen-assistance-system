package decision

import (
	"encoding/json"
	"testing"
)

func TestValidateDraftDefinitionAcceptsKhaiSinh(t *testing.T) {
	raw := []byte(`{
		"procedure_code": "dk_khai_sinh",
		"name": "Đăng ký khai sinh",
		"slots": {
			"noi_sinh": {"type": "string", "question": "Nơi sinh?"}
		},
		"required_slots": ["noi_sinh"],
		"guidance": {"summary": "x", "checklist": ["Giấy chứng sinh"]}
	}`)
	got := ValidateDraftDefinition(raw, "dk_khai_sinh")
	if !got.SchemaValid {
		t.Fatalf("issues=%v", got.Issues)
	}
	if got.SemanticApproved {
		t.Fatal("semantic approval must stay false")
	}
}

func TestValidateDraftDefinitionRejectsBadSlotType(t *testing.T) {
	raw := []byte(`{
		"procedure_code": "dk_khai_sinh",
		"name": "Đăng ký khai sinh",
		"slots": {"x": {"type": "date", "question": "?"}},
		"required_slots": []
	}`)
	got := ValidateDraftDefinition(raw, "dk_khai_sinh")
	if got.SchemaValid {
		t.Fatal("expected invalid")
	}
}

func TestValidateDraftDefinitionRejectsBrokenCondition(t *testing.T) {
	raw := []byte(`{
		"procedure_code": "dk_khai_sinh",
		"name": "Đăng ký khai sinh",
		"slots": {"a": {"type": "boolean", "question": "?"}},
		"conditional_slots": [{"when": {"slot": "missing", "equals": true}, "require": ["a"]}]
	}`)
	got := ValidateDraftDefinition(raw, "dk_khai_sinh")
	if got.SchemaValid {
		t.Fatal("expected invalid")
	}
}

func TestValidateDraftDefinitionRejectsSupportMap(t *testing.T) {
	raw := []byte(`{
		"procedure_code": "dk_khai_sinh",
		"name": "Đăng ký khai sinh",
		"slots": {"a": {"type": "string", "question": "?"}},
		"support_map": {"a": ["doc"]}
	}`)
	got := ValidateDraftDefinition(raw, "dk_khai_sinh")
	if got.SchemaValid {
		t.Fatal("expected forbidden key")
	}
}

func TestParseDefinitionDoesNotActivate(t *testing.T) {
	def := khaiSinh()
	if def.ProcedureCode != "dk_khai_sinh" {
		t.Fatal(def.ProcedureCode)
	}
}

func TestEmptyDefinitionIsNotProcedurallyReady(t *testing.T) {
	raw := []byte(`{
		"procedure_code": "dk_khai_sinh",
		"name": "Đăng ký khai sinh",
		"intent_examples": ["ui"],
		"slots": {},
		"required_slots": [],
		"guidance": {"summary": "", "checklist": []}
	}`)
	got := ValidateDraftDefinition(raw, "dk_khai_sinh")
	if !got.SchemaValid {
		t.Fatalf("empty slots can be structurally valid, issues=%v", got.Issues)
	}
	if got.ProceduralReady {
		t.Fatal("empty definition must not be procedurally ready")
	}
}

func TestMinimalUsableDefinitionIsProcedurallyReady(t *testing.T) {
	raw := []byte(`{
		"procedure_code": "tiny_proc",
		"name": "Thủ tục tối thiểu",
		"slots": {"noi_sinh": {"type": "string", "question": "Nơi sinh?"}},
		"required_slots": ["noi_sinh"],
		"guidance": {"summary": "Nộp tại UBND"}
	}`)
	got := ValidateDraftDefinition(raw, "tiny_proc")
	if !got.SchemaValid || !got.ProceduralReady {
		t.Fatalf("minimal usable def must pass, issues=%v", got.Issues)
	}
}

func TestDraftCitationSourceConflict(t *testing.T) {
	mk := func(docID, documentID string) []byte {
		raw, _ := json.Marshal(map[string]any{
			"procedure_code": "khai_sinh_v1", "name": "Đăng ký khai sinh", "intent_examples": []string{"khai sinh"},
			"slots":          map[string]any{"hinh_thuc_nop": map[string]any{"type": "string", "question": "Hình thức nộp?"}},
			"required_slots": []string{}, "conditional_slots": []any{},
			"guidance":  map[string]any{"summary": "s", "checklist": []string{"a"}, "where_to_submit": "UBND"},
			"citations": []any{map[string]any{"doc_id": docID, "document_id": documentID, "title": "Quyết định 163", "source_type": "ai_draft_unapproved"}},
		})
		return raw
	}
	has := func(raw []byte) bool {
		for _, is := range ValidateDraftDefinition(raw, "khai_sinh_v1").Issues {
			if is.Code == "CITATION_SOURCE_CONFLICT" && is.Severity == "error" {
				return true
			}
		}
		return false
	}
	if !has(mk("document-a", "document-b")) {
		t.Fatal("doc_id and document_id naming different documents must be an error")
	}
	for _, ok := range [][2]string{
		{"document-a", "DOCUMENT-A"}, {"document-a", ""}, {"", "document-b"},
		{"8939d98f-ce2c-487f-9f28-a53510c29271", "8939D98FCE2C487F9F28A53510C29271"},
		{"{8939d98f-ce2c-487f-9f28-a53510c29271}", "urn:uuid:8939d98f-ce2c-487f-9f28-a53510c29271"},
	} {
		if has(mk(ok[0], ok[1])) {
			t.Fatalf("%v is not a conflict", ok)
		}
	}
}

func TestNormalizeSourceID(t *testing.T) {
	const canon = "8939d98f-ce2c-487f-9f28-a53510c29271"
	for _, in := range []string{
		canon,
		"  8939D98F-CE2C-487F-9F28-A53510C29271 ",
		"8939d98fce2c487f9f28a53510c29271",
		"8939D98FCE2C487F9F28A53510C29271",
		"{8939d98f-ce2c-487f-9f28-a53510c29271}",
		"{8939D98F-CE2C-487F-9F28-A53510C29271}",
		"urn:uuid:8939d98f-ce2c-487f-9f28-a53510c29271",
		"urn:uuid:8939D98F-CE2C-487F-9F28-A53510C29271",
		"URN:UUID:8939D98F-CE2C-487F-9F28-A53510C29271",
	} {
		if got := NormalizeSourceID(in); got != canon {
			t.Errorf("NormalizeSourceID(%q) = %q, want %q", in, got, canon)
		}
	}
	for in, want := range map[string]string{
		" QĐ-163/BTP ":                          "qđ-163/btp",
		"":                                      "",
		"8939d98f-ce2c-487f-9f28-a53510c2927":   "8939d98f-ce2c-487f-9f28-a53510c2927",   // one digit short: not a UUID
		"8939d98f-ce2c-487f-9f28-a53510c29271x": "8939d98f-ce2c-487f-9f28-a53510c29271x", // trailing junk: not a UUID
	} {
		if got := NormalizeSourceID(in); got != want {
			t.Errorf("NormalizeSourceID(%q) = %q, want %q", in, got, want)
		}
	}
}
