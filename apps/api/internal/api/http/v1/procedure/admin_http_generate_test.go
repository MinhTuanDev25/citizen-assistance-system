package procedure

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/auth"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/drafting"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/middleware"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/repository"
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestAdminHTTPGenerateGetUnauthorized(t *testing.T) {
	dbURL := os.Getenv("DATABASE_URL")
	if dbURL == "" {
		t.Skip("DATABASE_URL required")
	}
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, dbURL)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(pool.Close)

	procID := uuid.MustParse("be19579e-6edd-4ae2-8f10-a227eeb0510e")
	docID := uuid.MustParse("60b677df-acc0-4198-a3b6-a919c01482a1")
	genID := uuid.MustParse("e2ad49da-4db7-4e18-90e3-c2f0bb358940")
	var activeBefore *uuid.UUID
	if err := pool.QueryRow(ctx, `SELECT active_version_id FROM procedures WHERE id=$1`, procID).Scan(&activeBefore); err != nil {
		t.Fatal(err)
	}

	source := &drafting.SourceStore{Pool: pool}
	snap, err := source.Resolve(ctx, "xa_chu_se", procID, docID, genID, 10, 21)
	if err != nil {
		t.Fatal(err)
	}
	quote := snap.Excerpts[0].Content
	if len(quote) > 80 {
		quote = quote[:80]
	}
	definition := map[string]any{
		"procedure_code":  "dk_khai_sinh",
		"name":            "Đăng ký khai sinh",
		"intent_examples": []string{"Đăng ký khai sinh"},
		"slots":           map[string]any{"noi_sinh": map[string]any{"type": "string", "question": "Nơi sinh?"}},
		"required_slots":  []string{"noi_sinh"},
		"guidance":        map[string]any{"summary": "Nộp tại UBND cấp xã", "checklist": []string{"Giấy chứng sinh"}},
		"citations":       []map[string]any{{"doc_id": docID.String(), "title": "QĐ 163", "source_type": "ai_draft_unapproved"}},
	}
	defRaw, _ := json.Marshal(definition)
	payload, _ := json.Marshal(map[string]any{
		"schema_version": "draft_definition.v1",
		"request_id":     uuid.NewString(),
		"procedure_code": "dk_khai_sinh",
		"definition":     json.RawMessage(defRaw),
		"evidence": []map[string]any{{
			"proposal_id": "slot:noi_sinh", "kind": "slot",
			"excerpt_id": snap.Excerpts[0].ExcerptID.String(),
			"page":       snap.Excerpts[0].PageStart, "quotation": quote,
		}, {
			"proposal_id": "required:noi_sinh", "kind": "required_slot",
			"excerpt_id": snap.Excerpts[0].ExcerptID.String(),
			"page":       snap.Excerpts[0].PageStart, "quotation": quote,
		}, {
			"proposal_id": "checklist:0", "kind": "checklist",
			"excerpt_id": snap.Excerpts[0].ExcerptID.String(),
			"page":       snap.Excerpts[0].PageStart, "quotation": quote,
		}, {
			"proposal_id": "guidance:summary", "kind": "guidance",
			"excerpt_id": snap.Excerpts[0].ExcerptID.String(),
			"page":       snap.Excerpts[0].PageStart, "quotation": quote,
		}},
		"warnings": []string{}, "missing_requirements": []string{},
		"validation": map[string]any{"schema_valid": true, "grounding_verified": false, "semantic_approved": false, "issues": []any{}, "readiness": "needs_review"},
		"provider":   "mock", "model": "http-e2e-mock", "prompt_version": "draft.prompt.v2", "prompt_hash": "abc",
		"usage": map[string]any{"total_token_count": 1},
	})
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer test-service-token" {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(payload)
	}))
	t.Cleanup(upstream.Close)

	client := drafting.NewClient(upstream.URL, 5*time.Second)
	client.ServiceToken = "test-service-token"
	svc := &drafting.Service{
		XAID: "xa_chu_se", Source: source, Drafts: &repository.DraftRepo{Pool: pool}, Client: client,
	}
	gin.SetMode(gin.TestMode)
	tokens := &auth.TokenService{Secret: []byte("phase45-http-e2e-secret-key"), TTL: time.Hour}
	actor := uuid.MustParse("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
	adminTok, _, err := tokens.Issue(actor, auth.RoleAdmin, "admin@chuse.vn", "Admin")
	if err != nil {
		t.Fatal(err)
	}
	h := NewHandler(nil, "xa_chu_se", []string{"ho_tich_chung_thuc"})
	h.Drafting = svc
	r := gin.New()
	g := r.Group("/api/v1/admin/procedures")
	g.Use(middleware.RequireAdmin(tokens))
	g.POST("/:procedureId/ai-drafts", h.GenerateAIDraft)
	d := r.Group("/api/v1/admin/procedure-drafts")
	d.Use(middleware.RequireAdmin(tokens))
	d.GET("/:draftId", h.GetDraft)

	unauth := httptest.NewRequest(http.MethodPost, "/api/v1/admin/procedures/"+procID.String()+"/ai-drafts", bytes.NewBufferString(`{"document_id":"`+docID.String()+`","generation_id":"`+genID.String()+`"}`))
	unauth.Header.Set("X-Request-ID", uuid.NewString())
	w0 := httptest.NewRecorder()
	r.ServeHTTP(w0, unauth)
	if w0.Code != http.StatusUnauthorized {
		t.Fatalf("unauth=%d", w0.Code)
	}

	req := httptest.NewRequest(http.MethodPost, "/api/v1/admin/procedures/"+procID.String()+"/ai-drafts", bytes.NewBufferString(`{"document_id":"`+docID.String()+`","generation_id":"`+genID.String()+`","source_page_start":10,"source_page_end":21}`))
	req.Header.Set("Authorization", "Bearer "+adminTok)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Request-ID", uuid.NewString())
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != http.StatusCreated {
		t.Fatalf("generate %d %s", w.Code, w.Body.String())
	}
	var env struct {
		Data drafting.DraftView `json:"data"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &env); err != nil {
		t.Fatal(err)
	}
	if env.Data.ID == uuid.Nil || env.Data.ProcedureCode != "dk_khai_sinh" || env.Data.Status != "DRAFT" {
		t.Fatalf("view %+v", env.Data)
	}
	if env.Data.Validation.SemanticApproved {
		t.Fatal("http generate must not auto-approve")
	}

	get := httptest.NewRequest(http.MethodGet, "/api/v1/admin/procedure-drafts/"+env.Data.ID.String(), nil)
	get.Header.Set("Authorization", "Bearer "+adminTok)
	get.Header.Set("X-Request-ID", uuid.NewString())
	w2 := httptest.NewRecorder()
	r.ServeHTTP(w2, get)
	if w2.Code != http.StatusOK {
		t.Fatalf("get %d %s", w2.Code, w2.Body.String())
	}

	var activeAfter *uuid.UUID
	if err := pool.QueryRow(ctx, `SELECT active_version_id FROM procedures WHERE id=$1`, procID).Scan(&activeAfter); err != nil {
		t.Fatal(err)
	}
	if activeBefore == nil || activeAfter == nil || *activeBefore != *activeAfter {
		t.Fatalf("ACTIVE changed %v -> %v", activeBefore, activeAfter)
	}
}
