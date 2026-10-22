package procedure

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/auth"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/decision"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/drafting"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/middleware"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/repository"
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestLiveGeminiDraftPersistRetrieveHTTP(t *testing.T) {
	path := os.Getenv("LIVE_DRAFT_JSON")
	if path == "" {
		t.Skip("LIVE_DRAFT_JSON not set")
	}
	dbURL := os.Getenv("DATABASE_URL")
	if dbURL == "" {
		t.Fatal("DATABASE_URL required")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var live map[string]any
	if err := json.Unmarshal(raw, &live); err != nil {
		t.Fatal(err)
	}
	defBytes, err := json.Marshal(live["definition"])
	if err != nil {
		t.Fatal(err)
	}
	goVal := decision.ValidateDraftDefinition(defBytes, "dk_khai_sinh")
	if live["procedure_code"] != "dk_khai_sinh" {
		t.Fatalf("procedure %v", live["procedure_code"])
	}

	ctx := context.Background()
	pool, err := pgxpool.New(ctx, dbURL)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(pool.Close)
	procID := uuid.MustParse("be19579e-6edd-4ae2-8f10-a227eeb0510e")
	docID := uuid.MustParse("60b677df-acc0-4198-a3b6-a919c01482a1")
	var activeBefore *uuid.UUID
	if err := pool.QueryRow(ctx, `SELECT active_version_id FROM procedures WHERE id=$1`, procID).Scan(&activeBefore); err != nil {
		t.Fatal(err)
	}

	evidence, _ := json.Marshal(live["evidence"])
	var evidenceItems []drafting.EvidenceItem
	_ = json.Unmarshal(evidence, &evidenceItems)
	payload, _ := json.Marshal(map[string]any{
		"schema_version":        "draft_definition.v1",
		"definition":            json.RawMessage(defBytes),
		"evidence":              evidenceItems,
		"warnings":              live["warnings"],
		"missing_requirements":  live["missing_requirements"],
		"model":                 live["model_used"],
		"provider":              live["provider"],
		"prompt_version":        live["prompt_version"],
		"prompt_hash":           live["prompt_hash"],
		"generation_id":         live["generation_id"],
		"document_id":           live["document_id"],
		"source_page_start":     10,
		"source_page_end":       21,
		"usage":                 live["usage"],
		"latency_ms":            live["latency_ms"],
		"historical_generation_note": "source generation belongs to archived version 1.0.2; Wave 1 does not attach it to a new version",
	})
	validationRaw, _ := json.Marshal(goVal)
	actor := uuid.MustParse("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
	repo := &repository.DraftRepo{Pool: pool}
	row, err := repo.Insert(ctx, repository.ProcedureDraft{
		DocumentID:       docID,
		ProcedureID:      procID,
		DraftDefinition:  payload,
		ValidationResult: validationRaw,
		Status:           repository.DraftStatusDraft,
		CreatedBy:        &actor,
		UpdatedBy:        &actor,
	})
	if err != nil {
		t.Fatal(err)
	}

	svc := &drafting.Service{XAID: "xa_chu_se", Drafts: repo}
	got, err := svc.Get(ctx, row.ID)
	if err != nil {
		t.Fatal(err)
	}
	if got.ProcedureCode != "dk_khai_sinh" || got.Status != "DRAFT" {
		t.Fatalf("got %+v", got)
	}

	gin.SetMode(gin.TestMode)
	tokens := &auth.TokenService{Secret: []byte("phase45-live-secret-key"), TTL: time.Hour}
	adminTok, _, err := tokens.Issue(actor, auth.RoleAdmin, "admin@chuse.vn", "Admin")
	if err != nil {
		t.Fatal(err)
	}
	citizenTok, _, err := tokens.Issue(uuid.New(), auth.RoleCitizen, "c@test.vn", "Citizen")
	if err != nil {
		t.Fatal(err)
	}
	h := NewHandler(nil, "xa_chu_se", []string{"ho_tich_chung_thuc"})
	h.Drafting = svc
	r := gin.New()
	d := r.Group("/api/v1/admin/procedure-drafts")
	d.Use(middleware.RequireAdmin(tokens))
	d.GET("/:draftId", h.GetDraft)
	p := r.Group("/api/v1/admin/procedures")
	p.Use(middleware.RequireAdmin(tokens))
	p.GET("/:procedureId/drafts", h.ListDrafts)

	req := httptest.NewRequest(http.MethodGet, "/api/v1/admin/procedure-drafts/"+row.ID.String(), nil)
	req.Header.Set("Authorization", "Bearer "+adminTok)
	req.Header.Set("X-Request-ID", uuid.NewString())
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("admin get %d %s", w.Code, w.Body.String())
	}
	bad := httptest.NewRequest(http.MethodGet, "/api/v1/admin/procedure-drafts/"+row.ID.String(), nil)
	bad.Header.Set("Authorization", "Bearer "+citizenTok)
	bad.Header.Set("X-Request-ID", uuid.NewString())
	w2 := httptest.NewRecorder()
	r.ServeHTTP(w2, bad)
	if w2.Code != http.StatusForbidden {
		t.Fatalf("citizen get %d", w2.Code)
	}

	listReq := httptest.NewRequest(http.MethodGet, "/api/v1/admin/procedures/"+procID.String()+"/drafts", nil)
	listReq.Header.Set("Authorization", "Bearer "+adminTok)
	listReq.Header.Set("X-Request-ID", uuid.NewString())
	w3 := httptest.NewRecorder()
	r.ServeHTTP(w3, listReq)
	if w3.Code != http.StatusOK {
		t.Fatalf("list %d %s", w3.Code, w3.Body.String())
	}

	var activeAfter *uuid.UUID
	if err := pool.QueryRow(ctx, `SELECT active_version_id FROM procedures WHERE id=$1`, procID).Scan(&activeAfter); err != nil {
		t.Fatal(err)
	}
	if activeBefore == nil || activeAfter == nil || *activeBefore != *activeAfter {
		t.Fatalf("ACTIVE changed %v -> %v", activeBefore, activeAfter)
	}

	outDir := os.Getenv("WAVE1_OUT")
	if outDir == "" {
		return
	}
	_ = os.MkdirAll(outDir, 0o755)
	persist := map[string]any{
		"kind":                 "local_backend_e2e",
		"draft_id":             row.ID.String(),
		"procedure_id":         procID.String(),
		"document_id":          docID.String(),
		"status":               row.Status,
		"go_schema_valid":      goVal.SchemaValid,
		"go_semantic_approved": goVal.SemanticApproved,
		"go_readiness":         goVal.Readiness,
		"slot_count":           goVal.SlotCount,
		"checklist_count":      goVal.ChecklistCount,
		"active_version_id":    activeAfter.String(),
		"active_unchanged":     true,
		"admin_get_status":     http.StatusOK,
		"citizen_get_status":   w2.Code,
		"list_status":          w3.Code,
	}
	b, _ := json.MarshalIndent(persist, "", "  ")
	_ = os.WriteFile(filepath.Join(outDir, "draft_persistence_e2e.json"), b, 0o644)
	valOut := map[string]any{
		"kind":       "definition_validation",
		"go":         goVal,
		"python":     live["validation"],
		"model_used": live["model_used"],
	}
	vb, _ := json.MarshalIndent(valOut, "", "  ")
	_ = os.WriteFile(filepath.Join(outDir, "definition_validation_results.json"), vb, 0o644)
}
