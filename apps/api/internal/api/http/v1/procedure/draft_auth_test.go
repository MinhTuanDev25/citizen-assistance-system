package procedure

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/auth"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/middleware"
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

func draftEngine(t *testing.T) (*gin.Engine, string, string) {
	t.Helper()
	gin.SetMode(gin.TestMode)
	tokens := &auth.TokenService{Secret: []byte("phase45-test-secret-key"), TTL: time.Hour}
	adminID := uuid.New()
	adminTok, _, err := tokens.Issue(adminID, auth.RoleAdmin, "admin@test.vn", "Admin")
	if err != nil {
		t.Fatal(err)
	}
	citizenTok, _, err := tokens.Issue(uuid.New(), auth.RoleCitizen, "c@test.vn", "Citizen")
	if err != nil {
		t.Fatal(err)
	}
	h := NewHandler(nil, "xa_chu_se", []string{"ho_tich_chung_thuc"})
	r := gin.New()
	r.Use(middleware.RequestID())
	g := r.Group("/api/v1/admin/procedures")
	g.Use(middleware.RequireAdmin(tokens))
	g.POST("/:procedureId/ai-drafts", h.GenerateAIDraft)
	g.GET("/:procedureId/drafts", h.ListDrafts)
	d := r.Group("/api/v1/admin/procedure-drafts")
	d.Use(middleware.RequireAdmin(tokens))
	d.GET("/:draftId", h.GetDraft)
	d.POST("/:draftId/approve", h.ApproveDraft)
	d.POST("/:draftId/activate", h.ActivateDraft)
	return r, adminTok, citizenTok
}

func TestCitizenCannotGenerateDraft(t *testing.T) {
	r, _, citizen := draftEngine(t)
	req := httptest.NewRequest(http.MethodPost, "/api/v1/admin/procedures/"+uuid.NewString()+"/ai-drafts", bytes.NewBufferString(`{"document_id":"`+uuid.NewString()+`"}`))
	req.Header.Set("Authorization", "Bearer "+citizen)
	req.Header.Set("X-Request-ID", uuid.NewString())
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != http.StatusForbidden {
		t.Fatalf("status=%d body=%s", w.Code, w.Body.String())
	}
}

func TestCitizenCannotGetDraft(t *testing.T) {
	r, _, citizen := draftEngine(t)
	req := httptest.NewRequest(http.MethodGet, "/api/v1/admin/procedure-drafts/"+uuid.NewString(), nil)
	req.Header.Set("Authorization", "Bearer "+citizen)
	req.Header.Set("X-Request-ID", uuid.NewString())
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != http.StatusForbidden {
		t.Fatalf("status=%d body=%s", w.Code, w.Body.String())
	}
}

func TestGenerateWithoutAuthIsUnauthorized(t *testing.T) {
	r, _, _ := draftEngine(t)
	req := httptest.NewRequest(http.MethodPost, "/api/v1/admin/procedures/"+uuid.NewString()+"/ai-drafts", bytes.NewBufferString(`{}`))
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != http.StatusUnauthorized {
		t.Fatalf("status=%d", w.Code)
	}
}

func TestCitizenCannotApproveOrActivate(t *testing.T) {
	r, _, citizen := draftEngine(t)
	for _, path := range []string{"/approve", "/activate"} {
		req := httptest.NewRequest(http.MethodPost, "/api/v1/admin/procedure-drafts/"+uuid.NewString()+path, bytes.NewBufferString(`{"confirm":true}`))
		req.Header.Set("Authorization", "Bearer "+citizen)
		req.Header.Set("X-Request-ID", uuid.NewString())
		w := httptest.NewRecorder()
		r.ServeHTTP(w, req)
		if w.Code != http.StatusForbidden {
			t.Fatalf("%s status=%d body=%s", path, w.Code, w.Body.String())
		}
	}
}

func TestAdminGenerateWithoutServiceIsUnavailable(t *testing.T) {
	r, token, _ := draftEngine(t)
	req := httptest.NewRequest(http.MethodPost, "/api/v1/admin/procedures/"+uuid.NewString()+"/ai-drafts", bytes.NewBufferString(`{"document_id":"`+uuid.NewString()+`"}`))
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("X-Request-ID", uuid.NewString())
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("status=%d body=%s", w.Code, w.Body.String())
	}
}
