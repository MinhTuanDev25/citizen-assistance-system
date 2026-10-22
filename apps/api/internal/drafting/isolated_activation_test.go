package drafting

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"
	"time"

	"github.com/google/uuid"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/activation"
)

func TestIsolatedCreateVersionDoesNotTouchHistoricalActiveOrG3(t *testing.T) {
	pool := testPool(t)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	prodProc := uuid.MustParse("be19579e-6edd-4ae2-8f10-a227eeb0510e")
	g3 := uuid.MustParse("e2ad49da-4db7-4e18-90e3-c2f0bb358940")
	var activeBefore *uuid.UUID
	var g3Version uuid.UUID
	var g3Status string
	if err := pool.QueryRow(ctx, `SELECT active_version_id FROM procedures WHERE id=$1`, prodProc).Scan(&activeBefore); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(ctx, `SELECT procedure_version_id, status FROM document_index_generations WHERE id=$1`, g3).Scan(&g3Version, &g3Status); err != nil {
		t.Fatal(err)
	}

	code := "w2_wave2_fixture"
	var fixtureID uuid.UUID
	err := pool.QueryRow(ctx, `
		INSERT INTO procedures (procedure_code, domain_id, name, xa_id)
		VALUES ($1, 'ho_tich_chung_thuc', 'Wave 2 fixture (không kích hoạt)', 'xa_chu_se')
		ON CONFLICT (xa_id, procedure_code) DO UPDATE SET name = EXCLUDED.name
		RETURNING id`, code).Scan(&fixtureID)
	if err != nil {
		t.Fatal(err)
	}

	def := map[string]any{
		"procedure_code":  code,
		"name":            "Wave 2 fixture (không kích hoạt)",
		"intent_examples": []string{"fixture"},
		"slots":           map[string]any{"noi_sinh": map[string]any{"type": "string", "question": "Nơi sinh?"}},
		"required_slots":  []string{"noi_sinh"},
		"guidance":        map[string]any{"summary": "Fixture", "checklist": []string{"Giấy tờ"}},
		"citations":       []map[string]any{{"doc_id": "doc", "title": "fixture", "source_type": "ai_draft_unapproved"}},
	}
	raw, _ := json.Marshal(def)
	svc := &activation.Service{Store: &activation.Store{Pool: pool}, XAID: "xa_chu_se"}
	actor := uuid.MustParse("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
	label := fmt.Sprintf("9.%d.0", time.Now().Unix()%100000)
	created, err := svc.CreateVersion(ctx, actor, uuid.New(), fixtureID, activation.CreateVersionRequest{
		Version:    label,
		Definition: raw,
		Confirm:    true,
	})
	if err != nil {
		t.Fatalf("create version: %v", err)
	}
	if created.Status != "APPROVED" {
		t.Fatalf("status=%s", created.Status)
	}

	_, err = svc.Activate(ctx, actor, uuid.New(), fixtureID, created.ID, nil)
	if err == nil {
		t.Fatal("activation without READY source must fail")
	}

	var activeProd *uuid.UUID
	if err := pool.QueryRow(ctx, `SELECT active_version_id FROM procedures WHERE id=$1`, prodProc).Scan(&activeProd); err != nil {
		t.Fatal(err)
	}
	if activeBefore == nil || activeProd == nil || *activeBefore != *activeProd {
		t.Fatalf("dk_khai_sinh ACTIVE changed: %v -> %v", activeBefore, activeProd)
	}
	var g3VersionAfter uuid.UUID
	var g3StatusAfter string
	if err := pool.QueryRow(ctx, `SELECT procedure_version_id, status FROM document_index_generations WHERE id=$1`, g3).Scan(&g3VersionAfter, &g3StatusAfter); err != nil {
		t.Fatal(err)
	}
	if g3VersionAfter != g3Version || g3StatusAfter != g3Status {
		t.Fatalf("G3 mutated: version %s->%s status %s->%s", g3Version, g3VersionAfter, g3Status, g3StatusAfter)
	}
	var fixtureActive *uuid.UUID
	if err := pool.QueryRow(ctx, `SELECT active_version_id FROM procedures WHERE id=$1`, fixtureID).Scan(&fixtureActive); err != nil {
		t.Fatal(err)
	}
	if fixtureActive != nil {
		t.Fatal("fixture must not become ACTIVE")
	}
}

func TestActivateDraftWithoutActivationService(t *testing.T) {
	now := time.Now()
	_, err := (&Service{}).Activate(context.Background(), uuid.New(), uuid.New(), uuid.New(), ActivateDraftRequest{
		Confirm: true, ExpectedUpdatedAt: &now,
	})
	if err == nil {
		t.Fatal("expected not configured")
	}
}
