package activation

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
)

func validDef() []byte {
	return []byte(`{
		"procedure_code": "chung_thuc_ban_sao",
		"name": "Chứng thực bản sao",
		"intent_examples": ["Chứng thực"],
		"slots": {},
		"required_slots": [],
		"guidance": {"summary": "x", "checklist": ["a"], "where_to_submit": "UBND"}
	}`)
}

func readySource(row versionRow) sourceRow {
	id := uuid.New()
	now := time.Now().UTC()
	return sourceRow{
		DocumentID: uuid.New(), Title: "PDF", RelationshipType: "SOURCE",
		IndexStatus: "READY", ValidityStatus: "VALID",
		XaID: row.XaID, DomainID: row.DomainID, ProcedureID: row.ProcedureID, ProcedureVersionID: row.ID,
		ActiveGenerationID: &id, GenerationStatus: "READY",
		ChunkCount: 2, VectorCount: 2, VectorDimension: 384, EmbeddingModelID: "e5-small",
		PublishedAt: &now,
	}
}

func approvedRow() versionRow {
	return versionRow{
		ID: uuid.New(), ProcedureID: uuid.New(), Version: "2.0.0", Status: "APPROVED",
		Definition: validDef(), ProcedureCode: "chung_thuc_ban_sao", ProcedureName: "Chứng thực bản sao",
		XaID: "xa_chu_se", DomainID: "ho_tich_chung_thuc",
	}
}

func TestEligibilityRequiresPublishedGeneration(t *testing.T) {
	row := approvedRow()
	cases := []struct {
		name   string
		mutate func(*sourceRow)
		code   string
	}{
		{name: "no source", mutate: nil, code: "NO_SOURCE_DOCUMENT"},
		{name: "uploaded", mutate: func(s *sourceRow) { s.IndexStatus = "UPLOADED"; s.ActiveGenerationID = nil }, code: "DOCUMENT_INDEX_UPLOADED"},
		{name: "processing", mutate: func(s *sourceRow) { s.IndexStatus = "PROCESSING"; s.ActiveGenerationID = nil }, code: "DOCUMENT_INDEX_PROCESSING"},
		{name: "failed", mutate: func(s *sourceRow) { s.IndexStatus = "FAILED"; s.ActiveGenerationID = nil }, code: "DOCUMENT_INDEX_FAILED"},
		{name: "missing generation", mutate: func(s *sourceRow) { s.ActiveGenerationID = nil }, code: "ACTIVE_GENERATION_MISSING"},
		{name: "staging generation", mutate: func(s *sourceRow) { s.GenerationStatus = "STAGING" }, code: "ACTIVE_GENERATION_MISSING"},
		{name: "failed generation", mutate: func(s *sourceRow) { s.GenerationStatus = "FAILED" }, code: "ACTIVE_GENERATION_MISSING"},
		{name: "mock ready", mutate: func(s *sourceRow) { s.PublishedAt = nil; s.ChunkCount = 0; s.VectorCount = 0 }, code: "MOCK_INDEX_NOT_PUBLISHED"},
		{name: "unlinked", mutate: func(s *sourceRow) { s.Unlinked = true }, code: "NO_SOURCE_DOCUMENT"},
		{name: "expired", mutate: func(s *sourceRow) { s.ValidityStatus = "EXPIRED" }, code: "DOCUMENT_NOT_CURRENT"},
		{name: "superseded doc", mutate: func(s *sourceRow) { s.ValidityStatus = "SUPERSEDED" }, code: "DOCUMENT_NOT_CURRENT"},
		{name: "pending failed index", mutate: func(s *sourceRow) {
			s.ValidityStatus = "PENDING"
			s.IndexStatus = "FAILED"
			s.ActiveGenerationID = nil
		}, code: "DOCUMENT_INDEX_FAILED"},
		{name: "pending missing generation", mutate: func(s *sourceRow) {
			s.ValidityStatus = "PENDING"
			s.ActiveGenerationID = nil
		}, code: "ACTIVE_GENERATION_MISSING"},
		{name: "wrong commune", mutate: func(s *sourceRow) { s.XaID = "other" }, code: "SOURCE_SCOPE_MISMATCH"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var sources []sourceRow
			if tc.mutate != nil {
				src := readySource(row)
				tc.mutate(&src)
				sources = []sourceRow{src}
			}
			el := evaluateEligibility(row, sources)
			if el.CanActivate {
				t.Fatal("should not activate")
			}
			if firstReason(el) != tc.code && !contains(el.Reasons, tc.code) {
				t.Fatalf("reasons=%v want %s", el.Reasons, tc.code)
			}
		})
	}
}

func TestEligibilityAllSourcesMustBeReady(t *testing.T) {
	row := approvedRow()
	good := readySource(row)
	bad := readySource(row)
	bad.IndexStatus = "PROCESSING"
	bad.ActiveGenerationID = nil
	el := evaluateEligibility(row, []sourceRow{good, bad})
	if el.CanActivate || !contains(el.Reasons, "PARTIAL_SOURCE_INDEX") {
		t.Fatalf("%+v", el)
	}
}

func TestEligibilityPass(t *testing.T) {
	row := approvedRow()
	el := evaluateEligibility(row, []sourceRow{readySource(row)})
	if !el.CanActivate {
		t.Fatalf("%+v", el)
	}
	row.Status = "INDEXING"
	if !evaluateEligibility(row, []sourceRow{readySource(row)}).CanActivate {
		t.Fatal("INDEXING with published source must be eligible")
	}
}

func TestPendingReadyGenerationIsEligible(t *testing.T) {
	row := approvedRow()
	src := readySource(row)
	src.ValidityStatus = "PENDING"
	if !evaluateEligibility(row, []sourceRow{src}).CanActivate {
		t.Fatal("PENDING + published READY generation must remain eligible")
	}
}

func TestDefinitionIdentity(t *testing.T) {
	row := approvedRow()
	src := readySource(row)
	if !evaluateEligibility(row, []sourceRow{src}).CanActivate {
		t.Fatal("matching identity must pass")
	}
	wrongCode := approvedRow()
	wrongCode.Definition = []byte(`{
		"procedure_code": "other_code",
		"name": "Chứng thực bản sao",
		"intent_examples": ["Chứng thực"],
		"slots": {},
		"required_slots": [],
		"guidance": {"summary": "x", "checklist": ["a"], "where_to_submit": "UBND"}
	}`)
	if firstReason(evaluateEligibility(wrongCode, []sourceRow{readySource(wrongCode)})) != "DEFINITION_IDENTITY_MISMATCH" {
		t.Fatal(evaluateEligibility(wrongCode, []sourceRow{readySource(wrongCode)}))
	}
	wrongXa := approvedRow()
	wrongXa.Definition = []byte(`{
		"procedure_code": "chung_thuc_ban_sao",
		"name": "Chứng thực bản sao",
		"xa_id": "other_xa",
		"intent_examples": ["Chứng thực"],
		"slots": {},
		"required_slots": [],
		"guidance": {"summary": "x", "checklist": ["a"], "where_to_submit": "UBND"}
	}`)
	if !contains(evaluateEligibility(wrongXa, []sourceRow{readySource(wrongXa)}).Reasons, "DEFINITION_IDENTITY_MISMATCH") {
		t.Fatal("wrong xa_id")
	}
	wrongDomain := approvedRow()
	wrongDomain.Definition = []byte(`{
		"procedure_code": "chung_thuc_ban_sao",
		"name": "Chứng thực bản sao",
		"domain": "dat_dai_nha_o",
		"intent_examples": ["Chứng thực"],
		"slots": {},
		"required_slots": [],
		"guidance": {"summary": "x", "checklist": ["a"], "where_to_submit": "UBND"}
	}`)
	if !contains(evaluateEligibility(wrongDomain, []sourceRow{readySource(wrongDomain)}).Reasons, "DEFINITION_IDENTITY_MISMATCH") {
		t.Fatal("wrong domain")
	}
	wrongVer := approvedRow()
	wrongVer.Definition = []byte(`{
		"procedure_code": "chung_thuc_ban_sao",
		"name": "Chứng thực bản sao",
		"version": "9.9.9",
		"intent_examples": ["Chứng thực"],
		"slots": {},
		"required_slots": [],
		"guidance": {"summary": "x", "checklist": ["a"], "where_to_submit": "UBND"}
	}`)
	if !contains(evaluateEligibility(wrongVer, []sourceRow{readySource(wrongVer)}).Reasons, "DEFINITION_IDENTITY_MISMATCH") {
		t.Fatal("wrong version")
	}
	wrongName := approvedRow()
	wrongName.Definition = []byte(`{
		"procedure_code": "chung_thuc_ban_sao",
		"name": "Tên khác",
		"intent_examples": ["Chứng thực"],
		"slots": {},
		"required_slots": [],
		"guidance": {"summary": "x", "checklist": ["a"], "where_to_submit": "UBND"}
	}`)
	if !contains(evaluateEligibility(wrongName, []sourceRow{readySource(wrongName)}).Reasons, "DEFINITION_IDENTITY_MISMATCH") {
		t.Fatal("wrong name")
	}
	legacy := approvedRow()
	legacy.Definition = validDef()
	if !evaluateEligibility(legacy, []sourceRow{readySource(legacy)}).CanActivate {
		t.Fatal("legacy without xa_id/domain/version must pass")
	}
	wrapped, err := wrapDefinition(validDef(), "chung_thuc_ban_sao", "Chứng thực bản sao", "ho_tich_chung_thuc", "xa_chu_se", "2.0.0", "APPROVED")
	if err != nil {
		t.Fatal(err)
	}
	row.Definition = wrapped
	if !evaluateEligibility(row, []sourceRow{readySource(row)}).CanActivate {
		t.Fatal("wrapDefinition candidate must pass")
	}
	if firstReason(evaluateEligibility(approvedRow(), nil)) == "DEFINITION_IDENTITY_MISMATCH" {
		t.Fatal("matching row without sources is NO_SOURCE, not identity")
	}
	spaced := approvedRow()
	spaced.ProcedureName = "Chứng thực   bản sao"
	if !evaluateEligibility(spaced, []sourceRow{readySource(spaced)}).CanActivate {
		t.Fatal("name whitespace must not reject")
	}
	malformed := approvedRow()
	malformed.Definition = []byte(`{not json`)
	if firstReason(evaluateEligibility(malformed, []sourceRow{readySource(malformed)})) != "INVALID_DEFINITION" {
		t.Fatal("malformed json")
	}
	badEngine := approvedRow()
	badEngine.Definition = []byte(`{"procedure_code":"chung_thuc_ban_sao","intent_examples":["x"]}`)
	if firstReason(evaluateEligibility(badEngine, []sourceRow{readySource(badEngine)})) != "INVALID_DEFINITION" {
		t.Fatal(evaluateEligibility(badEngine, []sourceRow{readySource(badEngine)}))
	}
}

func TestMapReasonCodes(t *testing.T) {
	if !errors.Is(mapReason(Eligibility{Reasons: []string{"INVALID_DEFINITION"}}), ErrInvalidDefinition) {
		t.Fatal("invalid definition")
	}
	if !errors.Is(mapReason(Eligibility{Reasons: []string{"VERSION_NOT_APPROVED"}}), ErrNotApproved) {
		t.Fatal("not approved")
	}
	if !errors.Is(mapReason(Eligibility{Reasons: []string{"NO_SOURCE_DOCUMENT"}}), ErrNotEligible) {
		t.Fatal("not eligible")
	}
	if !errors.Is(mapReason(Eligibility{Reasons: []string{"DEFINITION_IDENTITY_MISMATCH"}}), ErrNotEligible) {
		t.Fatal("identity")
	}
}

func TestEligibilityRejectsArchivedAndInvalidDefinition(t *testing.T) {
	row := versionRow{ID: uuid.New(), Status: "ARCHIVED", Definition: validDef(), XaID: "xa_chu_se"}
	if firstReason(evaluateEligibility(row, nil)) != "VERSION_ARCHIVED" {
		t.Fatal(evaluateEligibility(row, nil))
	}
	row.Status = "APPROVED"
	row.Definition = []byte(`{"procedure_code":""}`)
	if firstReason(evaluateEligibility(row, nil)) != "INVALID_DEFINITION" {
		t.Fatal(evaluateEligibility(row, nil))
	}
}

func TestEligibilityNeverActivatesSourceVersion(t *testing.T) {
	row := versionRow{ID: uuid.New(), Status: StatusSource, Definition: []byte(`{}`), XaID: "xa_chu_se"}
	ready := []sourceRow{{IndexStatus: "READY"}}
	el := evaluateEligibility(row, ready)
	if el.CanActivate || len(el.Reasons) != 1 || el.Reasons[0] != "SOURCE_VERSION" {
		t.Fatal(el)
	}
}

func TestCreateProcedureRejectsInvalidInputBeforeDB(t *testing.T) {
	svc := &Service{Store: &Store{}, XAID: "xa_chu_se"}
	for _, in := range []CreateProcedureRequest{
		{ProcedureCode: "DK", Name: "x", DomainID: "ho_tich_chung_thuc"},
		{ProcedureCode: "dk-khai-sinh", Name: "x", DomainID: "ho_tich_chung_thuc"},
		{ProcedureCode: "dk_khai_sinh", Name: " ", DomainID: "ho_tich_chung_thuc"},
		{ProcedureCode: "dk_khai_sinh", Name: "x", DomainID: ""},
	} {
		if _, err := svc.CreateProcedure(context.Background(), uuid.New(), uuid.New(), in); !errors.Is(err, ErrValidation) {
			t.Fatalf("%+v: want ErrValidation, got %v", in, err)
		}
	}
}

func TestCheckExpected(t *testing.T) {
	a, b := uuid.New(), uuid.New()
	if err := checkExpected(nil, nil); err != nil {
		t.Fatal(err)
	}
	if err := checkExpected(&a, &a); err != nil {
		t.Fatal(err)
	}
	if err := checkExpected(&a, &b); err != ErrStale {
		t.Fatal(err)
	}
	if err := checkExpected(&a, nil); err != ErrStale {
		t.Fatal(err)
	}
}

func TestWrapDefinitionRewritesIdentity(t *testing.T) {
	out, err := wrapDefinition(validDef(), "chung_thuc_ban_sao", "Chứng thực bản sao", "ho_tich_chung_thuc", "xa_chu_se", "2.0.0", "APPROVED")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(out), `"version":"2.0.0"`) && !strings.Contains(string(out), `"version": "2.0.0"`) {
		t.Fatal(string(out))
	}
}

func contains(in []string, want string) bool {
	for _, s := range in {
		if s == want {
			return true
		}
	}
	return false
}
