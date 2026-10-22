package httpserver

import (
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/config"
)

func TestIngestionDisabledDoesNotMountDocumentRoutes(t *testing.T) {
	eng := New(slog.New(slog.NewTextHandler(io.Discard, nil)), nil, config.Config{
		XAID: "xa_chu_se", JWTSecret: "integration-test-jwt-secret-key", JWTExpireHours: 24,
	})
	req := httptest.NewRequest(http.MethodPost, "/api/v1/admin/documents", strings.NewReader("x"))
	w := httptest.NewRecorder()
	eng.ServeHTTP(w, req)
	if w.Code != http.StatusNotFound {
		t.Fatalf("disabled ingestion mounted a route: %d", w.Code)
	}
}
