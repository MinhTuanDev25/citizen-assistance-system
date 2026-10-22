package drafting

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/google/uuid"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/repository"
)

func TestLiveHTTPGeneratePersistRetrieve(t *testing.T) {
	base := os.Getenv("LIVE_AI_URL")
	token := os.Getenv("AI_SERVICE_TOKEN")
	if base == "" || token == "" {
		t.Skip("LIVE_AI_URL and AI_SERVICE_TOKEN required")
	}
	pool := testPool(t)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	procID := uuid.MustParse("be19579e-6edd-4ae2-8f10-a227eeb0510e")
	docID := uuid.MustParse("60b677df-acc0-4198-a3b6-a919c01482a1")
	genID := uuid.MustParse("e2ad49da-4db7-4e18-90e3-c2f0bb358940")
	var activeBefore *uuid.UUID
	if err := pool.QueryRow(ctx, `SELECT active_version_id FROM procedures WHERE id=$1`, procID).Scan(&activeBefore); err != nil {
		t.Fatal(err)
	}
	client := NewClient(base, 90*time.Second)
	client.ServiceToken = token
	svc := &Service{
		XAID:   "xa_chu_se",
		Source: &SourceStore{Pool: pool},
		Drafts: &repository.DraftRepo{Pool: pool},
		Client: client,
	}
	actor := uuid.MustParse("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
	view, err := svc.Generate(ctx, actor, uuid.New(), procID, GenerateRequest{
		DocumentID:      docID,
		GenerationID:    genID,
		SourcePageStart: 10,
		SourcePageEnd:   21,
	})
	if err != nil {
		t.Fatalf("live generate: %v", err)
	}
	got, err := svc.Get(ctx, view.ID)
	if err != nil {
		t.Fatal(err)
	}
	if got.ID != view.ID || got.Status != "DRAFT" {
		t.Fatalf("retrieve %+v", got)
	}
	var activeAfter *uuid.UUID
	if err := pool.QueryRow(ctx, `SELECT active_version_id FROM procedures WHERE id=$1`, procID).Scan(&activeAfter); err != nil {
		t.Fatal(err)
	}
	if *activeBefore != *activeAfter {
		t.Fatalf("ACTIVE changed")
	}
	outDir := os.Getenv("WAVE1_OUT")
	if outDir == "" {
		return
	}
	_ = os.MkdirAll(outDir, 0o755)
	record := map[string]any{
		"kind":              "local_backend_e2e",
		"path":              "Go Source.Resolve -> FastAPI /v1/draft_definition -> Gemini -> procedure_drafts -> Get",
		"draft_id":          view.ID.String(),
		"procedure_code":    view.ProcedureCode,
		"status":            view.Status,
		"model":             view.Model,
		"provider":          view.Provider,
		"prompt_version":    view.PromptVersion,
		"schema_valid":      view.Validation.SchemaValid,
		"grounding":         view.Validation.GroundingVerified,
		"semantic_approved": view.Validation.SemanticApproved,
		"readiness":         view.Validation.Readiness,
		"active_unchanged":  true,
		"usage":             view.Usage,
		"latency_ms":        view.LatencyMS,
	}
	b, _ := json.MarshalIndent(record, "", "  ")
	_ = os.WriteFile(filepath.Join(outDir, "live_http_generate_e2e.json"), b, 0o644)
}
