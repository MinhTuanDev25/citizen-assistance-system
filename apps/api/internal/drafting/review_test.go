package drafting

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/google/uuid"
)

func TestApprovalBlockedWhenCoverageIncomplete(t *testing.T) {
	ex := testExcerpt()
	def := testDefinition(nil)
	view := DraftView{
		ProcedureCode: "dk_khai_sinh",
		Definition:    def,
		Evidence:      []EvidenceItem{validItem("slot:noi_sinh", "giấy chứng sinh")},
	}
	snap := SourceSnapshot{Excerpts: []Excerpt{ex}}
	revalidate(&view, snap, "dk_khai_sinh", "")
	if err := approvalGate(view); err == nil {
		t.Fatal("incomplete coverage must block approval")
	}
}

func TestApprovalRequiresExplicitAccept(t *testing.T) {
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
		item := validItem(id, quote)
		item.ReviewStatus = ReviewPending
		items = append(items, item)
	}
	view := DraftView{ProcedureCode: "dk_khai_sinh", Definition: testDefinition(nil), Evidence: items}
	revalidate(&view, SourceSnapshot{Excerpts: []Excerpt{ex}}, "dk_khai_sinh", "")
	if !view.Validation.GroundingVerified {
		t.Fatalf("expected grounding, issues=%v", view.Validation.Issues)
	}
	if err := approvalGate(view); err == nil {
		t.Fatal("pending reviews must block semantic approval")
	}
	decisions := make([]ReviewDecision, 0, len(ids))
	for _, id := range ids {
		decisions = append(decisions, ReviewDecision{ProposalID: id, Decision: ReviewAccepted})
	}
	if err := applyReviewDecisions(&view, decisions); err != nil {
		t.Fatal(err)
	}
	if err := approvalGate(view); err != nil {
		t.Fatal(err)
	}
}

func TestIdentityMismatchBlocksApproval(t *testing.T) {
	raw := testDefinition(map[string]any{"procedure_code": "other_proc"})
	locked, original, mismatch, err := lockProcedureIdentity(raw, "dk_khai_sinh", "Đăng ký khai sinh")
	if err != nil {
		t.Fatal(err)
	}
	if !mismatch || original != "other_proc" {
		t.Fatalf("mismatch=%v original=%s", mismatch, original)
	}
	var obj map[string]any
	_ = json.Unmarshal(locked, &obj)
	if obj["procedure_code"] != "dk_khai_sinh" {
		t.Fatal("server must keep locked identity")
	}
	view := DraftView{Definition: locked, IdentityMismatch: true, ProcedureCode: "dk_khai_sinh"}
	if err := approvalGate(view); !errors.Is(err, ErrApprovalBlocked) {
		t.Fatalf("got %v", err)
	}
}

func TestMissingProcedureCodeIsMismatch(t *testing.T) {
	raw := testDefinition(nil)
	var obj map[string]any
	_ = json.Unmarshal(raw, &obj)
	delete(obj, "procedure_code")
	raw, _ = json.Marshal(obj)
	_, _, mismatch, err := lockProcedureIdentity(raw, "dk_khai_sinh", "Đăng ký khai sinh")
	if err != nil {
		t.Fatal(err)
	}
	if !mismatch {
		t.Fatal("missing procedure_code must be mismatch")
	}
}

func TestStalePatchFingerprint(t *testing.T) {
	ex := testExcerpt()
	items := []EvidenceItem{validItem("checklist:0", "giấy chứng sinh")}
	cov := EvaluateCoverage(testDefinition(nil), items, []Excerpt{ex}, SourceSnapshot{Excerpts: []Excerpt{ex}})
	edited := testDefinition(map[string]any{
		"guidance": map[string]any{
			"summary": "Nộp tại UBND cấp xã", "checklist": []string{"Đã sửa checklist", "Giấy tờ tùy thân"},
			"where_to_submit": "UBND cấp xã",
		},
	})
	stale := invalidateStaleEvidence(cov.Evidence, edited)
	if stale[0].ReviewStatus != ReviewStale {
		t.Fatalf("status=%s", stale[0].ReviewStatus)
	}
}

func TestExpectedUpdatedAtRequired(t *testing.T) {
	svc := &Service{}
	ctx := context.Background()
	_, err := svc.Patch(ctx, uuid.New(), uuid.New(), PatchRequest{})
	if err == nil {
		t.Fatal("expected updated_at required")
	}
	_, err = svc.Approve(ctx, uuid.New(), uuid.New(), ApproveRequest{Confirm: true})
	if err == nil {
		t.Fatal("expected updated_at required")
	}
}
