package procedure

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/activation"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/auth"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/middleware"
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

func adminEngine(t *testing.T) (*gin.Engine, string, string) {
	t.Helper()
	gin.SetMode(gin.TestMode)
	tokens := &auth.TokenService{Secret: []byte("phase4-test-secret-key"), TTL: time.Hour}
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
	g.GET("", h.ListAdmin)
	g.POST("/:procedureId/versions/:versionId/activate", h.Activate)
	return r, adminTok, citizenTok
}

func TestAdminActivateRequiresAdmin(t *testing.T) {
	r, _, citizen := adminEngine(t)
	proc := uuid.New()
	ver := uuid.New()
	req := httptest.NewRequest(http.MethodPost, "/api/v1/admin/procedures/"+proc.String()+"/versions/"+ver.String()+"/activate", bytes.NewBufferString(`{"expected_active_version_id":null}`))
	req.Header.Set("Authorization", "Bearer "+citizen)
	req.Header.Set("X-Request-ID", uuid.NewString())
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != http.StatusForbidden {
		t.Fatalf("status=%d body=%s", w.Code, w.Body.String())
	}
}

func TestAdminActivateRequiresAuth(t *testing.T) {
	r, _, _ := adminEngine(t)
	req := httptest.NewRequest(http.MethodPost, "/api/v1/admin/procedures/"+uuid.NewString()+"/versions/"+uuid.NewString()+"/activate", bytes.NewBufferString(`{"expected_active_version_id":null}`))
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != http.StatusUnauthorized {
		t.Fatalf("status=%d", w.Code)
	}
}

func TestAdminActivateRequiresRequestID(t *testing.T) {
	r, token, _ := adminEngine(t)
	req := httptest.NewRequest(http.MethodPost, "/api/v1/admin/procedures/"+uuid.NewString()+"/versions/"+uuid.NewString()+"/activate", bytes.NewBufferString(`{"expected_active_version_id":null}`))
	req.Header.Set("Authorization", "Bearer "+token)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != http.StatusUnprocessableEntity {
		t.Fatalf("status=%d body=%s", w.Code, w.Body.String())
	}
}

func TestAdminListRequiresAdmin(t *testing.T) {
	r, _, citizen := adminEngine(t)
	req := httptest.NewRequest(http.MethodGet, "/api/v1/admin/procedures", nil)
	req.Header.Set("Authorization", "Bearer "+citizen)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != http.StatusForbidden {
		t.Fatalf("status=%d body=%s", w.Code, w.Body.String())
	}
}

func TestAdminListWithoutServiceIsNotFound(t *testing.T) {
	r, token, _ := adminEngine(t)
	req := httptest.NewRequest(http.MethodGet, "/api/v1/admin/procedures", nil)
	req.Header.Set("Authorization", "Bearer "+token)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != http.StatusNotFound {
		t.Fatalf("status=%d", w.Code)
	}
}

func TestDecodeExpectedRequiresField(t *testing.T) {
	_, err := decodeExpected(bytes.NewBufferString(`{}`))
	if err == nil {
		t.Fatal("expected error")
	}
	id, err := decodeExpected(bytes.NewBufferString(`{"expected_active_version_id":null}`))
	if err != nil || id != nil {
		t.Fatalf("%v %v", id, err)
	}
	want := uuid.New()
	got, err := decodeExpected(bytes.NewBufferString(`{"expected_active_version_id":"` + want.String() + `"}`))
	if err != nil || got == nil || *got != want {
		t.Fatalf("%v %v", got, err)
	}
}

func TestFirstCodePrefersSpecificReason(t *testing.T) {
	if firstCode(fmt.Errorf("%w: MOCK_INDEX_NOT_PUBLISHED", activation.ErrNotEligible)) != "MOCK_INDEX_NOT_PUBLISHED" {
		t.Fatal(firstCode(fmt.Errorf("%w: MOCK_INDEX_NOT_PUBLISHED", activation.ErrNotEligible)))
	}
	if firstCode(fmt.Errorf("%w: DEFINITION_IDENTITY_MISMATCH", activation.ErrNotEligible)) != "DEFINITION_IDENTITY_MISMATCH" {
		t.Fatal(firstCode(fmt.Errorf("%w: DEFINITION_IDENTITY_MISMATCH", activation.ErrNotEligible)))
	}
}

func TestWriteActivationErrCodes(t *testing.T) {
	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.GET("/e", func(c *gin.Context) {
		writeActivationErr(c, activation.ErrStale)
	})
	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/e", nil))
	if w.Code != http.StatusConflict {
		t.Fatalf("status=%d", w.Code)
	}
	var env map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &env); err != nil {
		t.Fatal(err)
	}
	errObj := env["error"].(map[string]any)
	if errObj["code"] != "STALE_ACTIVE_VERSION" {
		t.Fatalf("%v", env)
	}
}
