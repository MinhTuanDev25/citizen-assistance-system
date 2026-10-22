package drafting

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/repository"
)

func testPool(t *testing.T) *pgxpool.Pool {
	t.Helper()
	url := os.Getenv("DATABASE_URL")
	if url == "" {
		t.Skip("DATABASE_URL required")
	}
	if !strings.Contains(url, "citizen_assistance_p4b_test") {
		t.Skip("requires citizen_assistance_p4b_test; refusing CI/shared DATABASE_URL")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	pool, err := pgxpool.New(ctx, url)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(pool.Close)
	return pool
}

func TestE2EGeneratePersistRetrieveNoActivation(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	procID := uuid.MustParse("be19579e-6edd-4ae2-8f10-a227eeb0510e")
	docID := uuid.MustParse("60b677df-acc0-4198-a3b6-a919c01482a1")
	genID := uuid.MustParse("e2ad49da-4db7-4e18-90e3-c2f0bb358940")

	var activeBefore *uuid.UUID
	if err := pool.QueryRow(ctx, `SELECT active_version_id FROM procedures WHERE id=$1`, procID).Scan(&activeBefore); err != nil {
		t.Fatal(err)
	}

	source := &SourceStore{Pool: pool}
	snap, err := source.Resolve(ctx, "xa_chu_se", procID, docID, genID, 10, 21)
	if err != nil {
		t.Fatalf("resolve source: %v", err)
	}
	if len(snap.Excerpts) == 0 {
		t.Fatal("no excerpts")
	}
	quote := snap.Excerpts[0].Content
	if len(quote) > 120 {
		quote = quote[:120]
	}
	definition := map[string]any{
		"procedure_code":  "dk_khai_sinh",
		"name":            "Đăng ký khai sinh",
		"intent_examples": []string{"Đăng ký khai sinh"},
		"slots": map[string]any{
			"noi_sinh": map[string]any{"type": "string", "question": "Nơi sinh?"},
		},
		"required_slots": []string{"noi_sinh"},
		"guidance": map[string]any{
			"summary":   "Nộp tại UBND cấp xã",
			"checklist": []string{"Giấy chứng sinh"},
		},
		"citations": []map[string]any{{
			"doc_id": docID.String(), "title": "Quyết định 163/QĐ-BTP", "source_type": "ai_draft_unapproved",
		}},
	}
	defRaw, _ := json.Marshal(definition)
	payload, _ := json.Marshal(map[string]any{
		"schema_version": "draft_definition.v1",
		"request_id":     uuid.NewString(),
		"procedure_code": "dk_khai_sinh",
		"definition":     json.RawMessage(defRaw),
		"evidence": []map[string]any{{
			"proposal_id":    "slot:noi_sinh",
			"kind":           "slot",
			"excerpt_id":     snap.Excerpts[0].ExcerptID.String(),
			"page":           snap.Excerpts[0].PageStart,
			"quotation":      quote,
			"review_warning": "",
		}, {
			"proposal_id":    "required:noi_sinh",
			"kind":           "required_slot",
			"excerpt_id":     snap.Excerpts[0].ExcerptID.String(),
			"page":           snap.Excerpts[0].PageStart,
			"quotation":      quote,
			"review_warning": "",
		}, {
			"proposal_id":    "checklist:0",
			"kind":           "checklist",
			"excerpt_id":     snap.Excerpts[0].ExcerptID.String(),
			"page":           snap.Excerpts[0].PageStart,
			"quotation":      quote,
			"review_warning": "",
		}, {
			"proposal_id":    "guidance:summary",
			"kind":           "guidance",
			"excerpt_id":     snap.Excerpts[0].ExcerptID.String(),
			"page":           snap.Excerpts[0].PageStart,
			"quotation":      quote,
			"review_warning": "",
		}},
		"warnings":             []string{},
		"missing_requirements": []string{},
		"validation": map[string]any{
			"schema_valid": true, "grounding_verified": true, "semantic_approved": false,
			"issues": []any{}, "readiness": "needs_review",
		},
		"provider": "mock", "model": "e2e-mock",
		"prompt_version": "draft.prompt.v1", "prompt_hash": "abc",
		"usage": map[string]any{"total_token_count": 1},
	})

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/draft_definition" {
			http.NotFound(w, r)
			return
		}
		if r.Header.Get("Authorization") != "Bearer test-service-token" {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(payload)
	}))
	t.Cleanup(upstream.Close)

	client := NewClient(upstream.URL, 5*time.Second)
	client.ServiceToken = "test-service-token"
	svc := &Service{
		XAID:   "xa_chu_se",
		Source: source,
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
		t.Fatalf("generate: %v", err)
	}
	if view.Status != "DRAFT" {
		t.Fatalf("status=%s", view.Status)
	}
	if view.Validation.SemanticApproved {
		t.Fatal("must not auto-approve")
	}
	got, err := svc.Get(ctx, view.ID)
	if err != nil {
		t.Fatal(err)
	}
	if got.ID != view.ID || got.ProcedureCode != "dk_khai_sinh" {
		t.Fatalf("retrieve mismatch %+v", got)
	}

	var activeAfter *uuid.UUID
	if err := pool.QueryRow(ctx, `SELECT active_version_id FROM procedures WHERE id=$1`, procID).Scan(&activeAfter); err != nil {
		t.Fatal(err)
	}
	if activeBefore == nil || activeAfter == nil || *activeBefore != *activeAfter {
		t.Fatalf("ACTIVE version changed: %v -> %v", activeBefore, activeAfter)
	}
	var status string
	if err := pool.QueryRow(ctx, `SELECT status FROM procedure_versions WHERE id=$1`, *activeAfter).Scan(&status); err != nil {
		t.Fatal(err)
	}
	if status != "ACTIVE" {
		t.Fatalf("active version status=%s", status)
	}
}

func TestUnsupportedClaimIsNotApproved(t *testing.T) {
	id := uuid.MustParse("11111111-1111-1111-1111-111111111111")
	snap := SourceSnapshot{Excerpts: []Excerpt{{
		ExcerptID: id, PageStart: 10, PageEnd: 10, Content: "Đăng ký khai sinh tại UBND cấp xã.",
	}}}
	cov := EvaluateCoverage(nil, []EvidenceItem{{
		ProposalID: "checklist:fee",
		Kind:       "checklist",
		ExcerptID:  id.String(),
		Page:       10,
		Quotation:  "nộp lệ phí một tỷ đồng bằng Bitcoin",
	}}, snap.Excerpts, snap)
	if cov.QuotesVerified {
		t.Fatal("unsupported quotation must fail")
	}
	if cov.Evidence[0].Status == "supporting" {
		t.Fatal("must not mark supporting")
	}
	found := false
	for _, issue := range cov.Issues {
		if issue.Code == "QUOTATION_NOT_FOUND" {
			found = true
		}
	}
	if !found {
		t.Fatalf("issues=%v", cov.Issues)
	}
}
