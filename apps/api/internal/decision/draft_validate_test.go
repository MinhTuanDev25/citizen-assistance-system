package decision

import "testing"

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
