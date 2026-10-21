package drafting

import (
	"encoding/json"
	"testing"

	"github.com/google/uuid"
)

func testExcerpt() Excerpt {
	return Excerpt{
		ExcerptID:    uuid.MustParse("11111111-1111-1111-1111-111111111111"),
		DocumentID:   uuid.MustParse("22222222-2222-2222-2222-222222222222"),
		GenerationID: uuid.MustParse("33333333-3333-3333-3333-333333333333"),
		PageStart:    10,
		PageEnd:      12,
		Content:      "Người đi đăng ký khai sinh nộp giấy chứng sinh tại UBND cấp xã. Lệ phí được miễn.",
		TextSHA256:   "abc",
	}
}

func testDefinition(extra map[string]any) []byte {
	def := map[string]any{
		"procedure_code":  "dk_khai_sinh",
		"name":            "Đăng ký khai sinh",
		"intent_examples": []string{"Đăng ký khai sinh"},
		"slots": map[string]any{
			"noi_sinh":           map[string]any{"type": "string", "question": "Nơi sinh?"},
			"co_giay_chung_sinh": map[string]any{"type": "boolean", "question": "Có giấy chứng sinh?"},
			"nguoi_di_dang_ky":   map[string]any{"type": "enum", "question": "Ai đi đăng ký?", "enum_values": []string{"cha", "me"}},
		},
		"required_slots": []string{"noi_sinh", "co_giay_chung_sinh"},
		"conditional_slots": []map[string]any{{
			"when":    map[string]any{"slot": "co_giay_chung_sinh", "equals": false},
			"require": []string{"nguoi_di_dang_ky"},
		}},
		"guidance": map[string]any{
			"summary":         "Nộp tại UBND cấp xã",
			"checklist":       []string{"Giấy chứng sinh", "Giấy tờ tùy thân"},
			"where_to_submit": "UBND cấp xã",
		},
		"citations": []map[string]any{{
			"doc_id": "22222222-2222-2222-2222-222222222222", "title": "QĐ 163", "source_type": "ai_draft_unapproved",
		}},
	}
	for k, v := range extra {
		def[k] = v
	}
	raw, _ := json.Marshal(def)
	return raw
}

func validItem(pid, quote string) EvidenceItem {
	ex := testExcerpt()
	return EvidenceItem{
		ProposalID: pid, Kind: "slot", ExcerptID: ex.ExcerptID.String(),
		Page: 10, Quotation: quote,
	}
}

func TestQuotationMatch(t *testing.T) {
	src := "Người đi đăng ký khai sinh nộp giấy chứng sinh tại UBND cấp xã."
	if !quotationInText("giấy chứng sinh", src) {
		t.Fatal("expected match")
	}
	if quotationInText("lệ phí một trăm triệu", src) {
		t.Fatal("unsupported claim must not match")
	}
}

func TestPartialEvidenceNotFullyGrounded(t *testing.T) {
	ex := testExcerpt()
	cov := EvaluateCoverage(testDefinition(nil), []EvidenceItem{
		validItem("slot:noi_sinh", "UBND cấp xã"),
	}, []Excerpt{ex}, SourceSnapshot{Excerpts: []Excerpt{ex}, DocumentID: ex.DocumentID, GenerationID: ex.GenerationID})
	if cov.GroundingVerified || cov.CoverageComplete {
		t.Fatalf("one valid quote must not establish global coverage: %+v", cov)
	}
	if cov.Missing == 0 {
		t.Fatal("expected missing material proposals")
	}
}

func TestFullCoveragePass(t *testing.T) {
	ex := testExcerpt()
	quote := "giấy chứng sinh"
	ids := []string{
		"slot:noi_sinh", "slot:co_giay_chung_sinh", "slot:nguoi_di_dang_ky",
		"required:noi_sinh", "required:co_giay_chung_sinh",
		"conditional:0", "checklist:0", "checklist:1",
		"guidance:summary", "guidance:where_to_submit",
	}
	items := make([]EvidenceItem, 0, len(ids))
	for _, id := range ids {
		items = append(items, validItem(id, quote))
	}
	cov := EvaluateCoverage(testDefinition(nil), items, []Excerpt{ex}, SourceSnapshot{Excerpts: []Excerpt{ex}})
	if !cov.CoverageComplete || !cov.QuotesVerified || !cov.GroundingVerified {
		t.Fatalf("expected full coverage, missing=%d invalid=%d issues=%v", cov.Missing, cov.Invalid, cov.Issues)
	}
}

func TestUnknownProposalIDFails(t *testing.T) {
	ex := testExcerpt()
	cov := EvaluateCoverage(testDefinition(nil), []EvidenceItem{
		validItem("slot:does_not_exist", "giấy chứng sinh"),
	}, []Excerpt{ex}, SourceSnapshot{Excerpts: []Excerpt{ex}})
	if cov.Unknown == 0 || cov.GroundingVerified {
		t.Fatal("unknown proposal must fail")
	}
}

func TestDuplicateProposalMappingFails(t *testing.T) {
	ex := testExcerpt()
	cov := EvaluateCoverage(testDefinition(nil), []EvidenceItem{
		validItem("slot:noi_sinh", "giấy chứng sinh"),
		validItem("slot:noi_sinh", "UBND cấp xã"),
	}, []Excerpt{ex}, SourceSnapshot{Excerpts: []Excerpt{ex}})
	if cov.Duplicates == 0 || cov.CoverageComplete {
		t.Fatal("duplicate mapping must fail")
	}
}

func TestMissingChecklistNotFullyGrounded(t *testing.T) {
	ex := testExcerpt()
	quote := "giấy chứng sinh"
	items := []EvidenceItem{
		validItem("slot:noi_sinh", quote),
		validItem("slot:co_giay_chung_sinh", quote),
		validItem("slot:nguoi_di_dang_ky", quote),
		validItem("required:noi_sinh", quote),
		validItem("required:co_giay_chung_sinh", quote),
		validItem("conditional:0", quote),
		validItem("checklist:0", quote),
		validItem("guidance:summary", quote),
		validItem("guidance:where_to_submit", quote),
	}
	cov := EvaluateCoverage(testDefinition(nil), items, []Excerpt{ex}, SourceSnapshot{Excerpts: []Excerpt{ex}})
	if cov.CoverageComplete {
		t.Fatal("missing checklist:1 must block coverage")
	}
	found := false
	for _, issue := range cov.Issues {
		if issue.ProposalID == "checklist:1" && issue.Code == "MISSING_EVIDENCE" {
			found = true
		}
	}
	if !found {
		t.Fatal("expected MISSING_EVIDENCE for checklist:1")
	}
}

func TestInvalidSourcePageAndHashFail(t *testing.T) {
	ex := testExcerpt()
	badPage := validItem("slot:noi_sinh", "giấy chứng sinh")
	badPage.Page = 99
	cov := EvaluateCoverage(testDefinition(nil), []EvidenceItem{badPage}, []Excerpt{ex}, SourceSnapshot{Excerpts: []Excerpt{ex}})
	if cov.QuotesVerified {
		t.Fatal("invalid page must fail quotes")
	}
	badHash := validItem("slot:noi_sinh", "giấy chứng sinh")
	badHash.SourceSHA256 = "deadbeef"
	cov = EvaluateCoverage(testDefinition(nil), []EvidenceItem{badHash}, []Excerpt{ex}, SourceSnapshot{
		Excerpts: []Excerpt{ex}, DocumentSHA256: "cafebabe",
	})
	if cov.QuotesVerified {
		t.Fatal("invalid source hash must fail")
	}
}

func TestStaleFingerprintAfterEdit(t *testing.T) {
	ex := testExcerpt()
	props, err := EnumerateMaterialProposals(testDefinition(nil))
	if err != nil {
		t.Fatal(err)
	}
	var fp string
	for _, p := range props {
		if p.ID == "checklist:0" {
			fp = p.Fingerprint
		}
	}
	item := validItem("checklist:0", "giấy chứng sinh")
	item.BoundFingerprint = fp
	edited := testDefinition(map[string]any{
		"guidance": map[string]any{
			"summary":         "Nộp tại UBND cấp xã",
			"checklist":       []string{"Giấy chứng sinh bản gốc", "Giấy tờ tùy thân"},
			"where_to_submit": "UBND cấp xã",
		},
	})
	cov := EvaluateCoverage(edited, []EvidenceItem{item}, []Excerpt{ex}, SourceSnapshot{Excerpts: []Excerpt{ex}})
	if cov.Stale == 0 {
		t.Fatal("edited checklist must stale previous evidence")
	}
	if cov.CoverageComplete {
		t.Fatal("stale evidence must not count as coverage")
	}
}

func TestLegacyPropSlotNormalization(t *testing.T) {
	ex := testExcerpt()
	item := validItem("prop_slot_co_giay_chung_sinh", "giấy chứng sinh")
	cov := EvaluateCoverage(testDefinition(nil), []EvidenceItem{item}, []Excerpt{ex}, SourceSnapshot{Excerpts: []Excerpt{ex}})
	if cov.Evidence[0].ProposalID != "slot:co_giay_chung_sinh" {
		t.Fatalf("canonical=%s", cov.Evidence[0].ProposalID)
	}
	if cov.Evidence[0].RawProposalID != "prop_slot_co_giay_chung_sinh" {
		t.Fatal("raw id must be preserved")
	}
}

func TestAmbiguousLegacyIDFailsSafely(t *testing.T) {
	ex := testExcerpt()
	cov := EvaluateCoverage(testDefinition(nil), []EvidenceItem{
		validItem("prop_something_weird", "giấy chứng sinh"),
	}, []Excerpt{ex}, SourceSnapshot{Excerpts: []Excerpt{ex}})
	if cov.Unknown == 0 || cov.CoverageComplete {
		t.Fatal("ambiguous legacy id must fail closed")
	}
}

func TestNoMaterialProposalsVacuousCoverage(t *testing.T) {
	def := map[string]any{
		"procedure_code":  "dk_khai_sinh",
		"name":            "Đăng ký khai sinh",
		"intent_examples": []string{"ui example only"},
		"slots":           map[string]any{},
		"required_slots":  []string{},
		"guidance":        map[string]any{"summary": "", "checklist": []string{}},
		"citations":       []map[string]any{},
	}
	raw, _ := json.Marshal(def)
	ex := testExcerpt()
	cov := EvaluateCoverage(raw, nil, []Excerpt{ex}, SourceSnapshot{Excerpts: []Excerpt{ex}})
	if !cov.CoverageComplete || cov.MaterialCount != 0 {
		t.Fatalf("zero material proposals must be documented vacuous coverage, got %+v", cov)
	}
	if !cov.QuotesVerified || !cov.GroundingVerified {
		t.Fatal("no quotes and no claims is quotes-verified + coverage-complete")
	}
}

func TestNonMaterialMetadataExcluded(t *testing.T) {
	props, err := EnumerateMaterialProposals(testDefinition(nil))
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range props {
		if p.Type == "intent" || p.ID == "name" || p.Type == "citation" {
			t.Fatalf("non-material leaked: %+v", p)
		}
	}
}

func TestInvalidExcerptID(t *testing.T) {
	ex := testExcerpt()
	cov := EvaluateCoverage(testDefinition(nil), []EvidenceItem{{
		ProposalID: "slot:noi_sinh",
		ExcerptID:  "99999999-9999-9999-9999-999999999999",
		Page:       10,
		Quotation:  "giấy chứng sinh",
	}}, []Excerpt{ex}, SourceSnapshot{Excerpts: []Excerpt{ex}})
	if cov.QuotesVerified {
		t.Fatal("expected invalid")
	}
	if cov.Evidence[0].Status != "source_reference_invalid" {
		t.Fatalf("status=%s", cov.Evidence[0].Status)
	}
}

func TestInvalidPage(t *testing.T) {
	ex := testExcerpt()
	cov := EvaluateCoverage(testDefinition(nil), []EvidenceItem{
		{ProposalID: "slot:noi_sinh", ExcerptID: ex.ExcerptID.String(), Page: 21, Quotation: "giấy chứng sinh"},
	}, []Excerpt{ex}, SourceSnapshot{Excerpts: []Excerpt{ex}})
	if cov.QuotesVerified {
		t.Fatal("expected invalid page")
	}
}
