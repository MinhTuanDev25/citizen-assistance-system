//go:build integration

package activation_test

import (
	"context"
	"errors"
	"testing"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/activation"
	"github.com/google/uuid"
)

func insertDraftRow(t *testing.T, fx fixture) uuid.UUID {
	t.Helper()
	id := uuid.New()
	if _, err := fx.pool.Exec(context.Background(), `
		INSERT INTO procedure_drafts (id, procedure_id, status, created_by, updated_by)
		VALUES ($1,$2,'APPROVED',$3,$3)`, id, fx.ProcedureID, actor()); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		ctx := context.Background()
		_, _ = fx.pool.Exec(ctx, `UPDATE procedure_versions SET source_draft_id = NULL WHERE source_draft_id = $1`, id)
		_, _ = fx.pool.Exec(ctx, `DELETE FROM procedure_drafts WHERE id = $1`, id)
	})
	return id
}

// The Admin created 2.0.0 and indexed its document; activating a draft fills
// 2.0.0 instead of adding 2.0.1.
func TestCreateVersionFillsAdminCreatedTarget(t *testing.T) {
	pool := requireP4DB(t)
	fx := newFixture(t, pool, true)
	ctx := context.Background()
	draft := insertDraftRow(t, fx)
	var code string
	if err := pool.QueryRow(ctx, `SELECT procedure_code FROM procedures WHERE id = $1`, fx.ProcedureID).Scan(&code); err != nil {
		t.Fatal(err)
	}
	def := validDef(code, code, "9.9.9")
	req := activation.CreateVersionRequest{Definition: def, Confirm: true, SourceDraftID: &draft, TargetVersionID: &fx.V2}
	reqID := uuid.New()

	out, err := fx.svc.CreateVersion(ctx, actor(), reqID, fx.ProcedureID, req)
	if err != nil {
		t.Fatal(err)
	}
	if out.ID != fx.V2 || out.Version != "2.0.0" {
		t.Fatalf("must fill the target, got %+v", out)
	}
	var n int
	var label, src string
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM procedure_versions WHERE procedure_id = $1`, fx.ProcedureID).Scan(&n); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(ctx, `SELECT definition->>'version', source_draft_id::text FROM procedure_versions WHERE id = $1`, fx.V2).Scan(&label, &src); err != nil {
		t.Fatal(err)
	}
	if n != 2 || label != "2.0.0" || src != draft.String() {
		t.Fatalf("versions=%d label=%s source=%s", n, label, src)
	}

	// Retry with the same request: same version, nothing new.
	again, err := fx.svc.CreateVersion(ctx, actor(), reqID, fx.ProcedureID, req)
	if err != nil || again.ID != fx.V2 {
		t.Fatalf("retry: %+v %v", again, err)
	}

	// Another draft cannot take a filled version, nor the active one.
	other := insertDraftRow(t, fx)
	for _, target := range []uuid.UUID{fx.V2, fx.V1} {
		_, err = fx.svc.CreateVersion(ctx, actor(), uuid.New(), fx.ProcedureID, activation.CreateVersionRequest{
			Definition: def, Confirm: true, SourceDraftID: &other, TargetVersionID: &target,
		})
		if !errors.Is(err, activation.ErrValidation) {
			t.Fatalf("target %s must be refused, got %v", target, err)
		}
	}
}
