package drafting

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/google/uuid"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/activation"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/repository"
)

func TestStableVersionLabelIsDeterministic(t *testing.T) {
	id := uuid.MustParse("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
	first := stableVersionLabel(id)
	second := stableVersionLabel(id)
	if first != second {
		t.Fatal("label must be stable")
	}
	other := uuid.MustParse("11111111-2222-3333-4444-555555555555")
	if stableVersionLabel(id) == stableVersionLabel(other) {
		t.Fatal("different drafts must not share a label")
	}
}

func TestDraftOpIDIsStablePerOperation(t *testing.T) {
	id := uuid.MustParse("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
	first := draftOpID(id, "index")
	second := draftOpID(id, "index")
	if first != second {
		t.Fatal("index request id must be stable")
	}
	if draftOpID(id, "index") == draftOpID(id, "link") {
		t.Fatal("different ops must not collide")
	}
}

func TestSourceIndexStateReadyProcessingFailed(t *testing.T) {
	ready := activation.VersionSummary{Documents: []activation.DocumentLink{{
		RelationshipType: "SOURCE", IndexStatus: "READY", GenerationStatus: "READY",
	}}}
	if sourceIndexState(ready) != "READY" {
		t.Fatal(sourceIndexState(ready))
	}
	run := activation.VersionSummary{Documents: []activation.DocumentLink{{
		RelationshipType: "SOURCE", IndexStatus: "PROCESSING",
	}}}
	if sourceIndexState(run) != "PROCESSING" {
		t.Fatal(sourceIndexState(run))
	}
	fail := activation.VersionSummary{Documents: []activation.DocumentLink{{
		RelationshipType: "SOURCE", IndexStatus: "FAILED",
	}}}
	if sourceIndexState(fail) != "FAILED" {
		t.Fatal(sourceIndexState(fail))
	}
}

func TestStoredPayloadKeepsOriginalProcedureCode(t *testing.T) {
	view := DraftView{
		ProcedureCode:         "dk_khai_sinh",
		OriginalProcedureCode: "wrong_code",
		IdentityMismatch:      true,
		Definition:            json.RawMessage(`{"procedure_code":"dk_khai_sinh","name":"x"}`),
	}
	raw, err := json.Marshal(storedPayloadFromView(view))
	if err != nil {
		t.Fatal(err)
	}
	var payload storedPayload
	if err := json.Unmarshal(raw, &payload); err != nil {
		t.Fatal(err)
	}
	if payload.OriginalProcedureCode != "wrong_code" {
		t.Fatalf("got %q", payload.OriginalProcedureCode)
	}
}

type failUpdateStatus struct {
	*repository.DraftRepo
	on    string
	err   error
	after int
	n     int
}

func (f *failUpdateStatus) UpdateStatus(ctx context.Context, xaID string, draftID, actor uuid.UUID, status string, definition, validation json.RawMessage, expectedUpdatedAt time.Time, allowedFrom []string) (repository.ProcedureDraft, error) {
	if f.on == "" || status == f.on {
		f.n++
		if f.after == 0 || f.n >= f.after {
			return repository.ProcedureDraft{}, f.err
		}
	}
	return f.DraftRepo.UpdateStatus(ctx, xaID, draftID, actor, status, definition, validation, expectedUpdatedAt, allowedFrom)
}

func TestPublishDraftKeepsApprovedViewWhenStatusUpdateFails(t *testing.T) {
	view := DraftView{
		Status:     "APPROVED",
		Definition: json.RawMessage(`{"procedure_code":"x","name":"x"}`),
	}
	svc := &Service{Drafts: &failUpdateStatus{
		on:  repository.DraftStatusPublished,
		err: errString("injected UpdateStatus failure"),
	}}
	err := svc.publishDraft(context.Background(), uuid.New(), uuid.New(), repository.ProcedureDraft{}, &view)
	if err == nil {
		t.Fatal("expected injected failure")
	}
	if view.Status != "APPROVED" {
		t.Fatalf("HTTP view must stay APPROVED, got %s", view.Status)
	}
}

func TestPersistApprovedPayloadReportsUpdateStatusError(t *testing.T) {
	svc := &Service{Drafts: &failUpdateStatus{
		on:  repository.DraftStatusApproved,
		err: errString("injected persist failure"),
	}}
	_, err := svc.persistApprovedPayload(context.Background(), uuid.New(), uuid.New(), time.Time{}, DraftView{})
	if err == nil || err.Error() != "injected persist failure" {
		t.Fatalf("got %v", err)
	}
}

func TestPersistedDraftViewOmitsUnsavedIndexIDs(t *testing.T) {
	reqID := uuid.New()
	jobID := uuid.New()
	view := DraftView{ID: uuid.New(), Status: "APPROVED", IndexRequestID: &reqID, IndexJobID: &jobID}
	out := persistedDraftView(context.Background(), &Service{}, view)
	if out.IndexRequestID != nil || out.IndexJobID != nil {
		t.Fatalf("unsaved index ids leaked: request=%v job=%v", out.IndexRequestID, out.IndexJobID)
	}
	if out.Status != "APPROVED" {
		t.Fatalf("status=%s", out.Status)
	}
}

func TestEmptyDefinitionBlocksApproval(t *testing.T) {
	raw := []byte(`{
		"procedure_code": "dk_khai_sinh",
		"name": "Đăng ký khai sinh",
		"intent_examples": ["ui"],
		"slots": {},
		"required_slots": [],
		"guidance": {"summary": "", "checklist": []},
		"citations": []
	}`)
	view := DraftView{ProcedureCode: "dk_khai_sinh", Definition: raw}
	revalidate(&view, SourceSnapshot{}, "dk_khai_sinh", "")
	if view.Validation.ProceduralReady {
		t.Fatal("empty def must not be ready")
	}
	if err := approvalGate(view); err == nil {
		t.Fatal("empty def must not be approved")
	}
}
