//go:build integration

package document_test

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"net/textproto"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/auth"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/config"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/httpserver"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/storage"
	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
)

func requireDB(t *testing.T) *pgxpool.Pool {
	t.Helper()
	if os.Getenv("CAS_INTEGRATION") != "1" {
		t.Fatal("CAS_INTEGRATION=1 required")
	}
	url := os.Getenv("DATABASE_URL")
	if url == "" {
		url = "postgres://cas:cas@127.0.0.1:5432/citizen_assistance?sslmode=disable"
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	pool, err := pgxpool.New(ctx, url)
	if err != nil {
		t.Fatal(err)
	}
	if err := pool.Ping(ctx); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(pool.Close)
	return pool
}

func engine(t *testing.T, pool *pgxpool.Pool, objects *storage.Memory) (http.Handler, string) {
	t.Helper()
	cfg := config.Config{
		Env: "local", XAID: "xa_chu_se", LogLevel: "error",
		CitizenDomainIDs:           []string{"ho_tich_chung_thuc"},
		JWTSecret:                  "integration-test-jwt-secret-key",
		JWTExpireHours:             24,
		AdminIngestionEnabled:      true,
		ObjectStorageBucket:        "cas-documents",
		ObjectStorageSSLConfigured: true,
		DocumentMaxBytes:           1 << 20,
	}
	tokens := &auth.TokenService{Secret: []byte(cfg.JWTSecret), TTL: time.Hour}
	adminID := uuid.MustParse("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
	tok, _, err := tokens.Issue(adminID, auth.RoleAdmin, "admin@chuse.vn", "Seed Admin")
	if err != nil {
		t.Fatal(err)
	}
	return httpserver.NewWithStore(slog.New(slog.NewTextHandler(io.Discard, nil)), pool, cfg, objects), tok
}

func upload(t *testing.T, eng http.Handler, token string, file []byte, fields map[string]string) *httptest.ResponseRecorder {
	t.Helper()
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)
	for k, v := range fields {
		if err := w.WriteField(k, v); err != nil {
			t.Fatal(err)
		}
	}
	hdr := make(textproto.MIMEHeader)
	hdr.Set("Content-Disposition", `form-data; name="file"; filename="a.pdf"`)
	hdr.Set("Content-Type", "application/pdf")
	part, err := w.CreatePart(hdr)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := part.Write(file); err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, "/api/v1/admin/documents", &buf)
	req.Header.Set("Content-Type", w.FormDataContentType())
	req.Header.Set("Authorization", "Bearer "+token)
	res := httptest.NewRecorder()
	eng.ServeHTTP(res, req)
	return res
}

func TestIntegrationUploadAuditDedupeAndScope(t *testing.T) {
	pool := requireDB(t)
	objects := storage.NewMemory()
	eng, token := engine(t, pool, objects)
	ctx := context.Background()
	pdf := append([]byte("%PDF-1.4\n"), []byte("integration-body")...)
	fields := map[string]string{
		"title": "Giay khai sinh", "domain_id": "ho_tich_chung_thuc",
		"xa_id": "xa_other", "effective_date": "2024-01-02",
	}
	res := upload(t, eng, token, pdf, fields)
	if res.Code != http.StatusCreated {
		t.Fatalf("upload %d %s", res.Code, res.Body.String())
	}
	var env struct {
		Data struct {
			ID       string `json:"id"`
			XAID     string `json:"xa_id"`
			Checksum string `json:"checksum"`
		} `json:"data"`
	}
	if err := json.Unmarshal(res.Body.Bytes(), &env); err != nil {
		t.Fatal(err)
	}
	if env.Data.XAID != "xa_chu_se" {
		t.Fatalf("client xa_id was trusted: %s", env.Data.XAID)
	}
	t.Cleanup(func() {
		_, _ = pool.Exec(context.Background(), `DELETE FROM audit_logs WHERE entity_id = $1`, env.Data.ID)
		_, _ = pool.Exec(context.Background(), `DELETE FROM documents WHERE id = $1`, env.Data.ID)
	})
	var payload string
	if err := pool.QueryRow(ctx, `SELECT payload::text FROM audit_logs WHERE entity_id = $1 AND action = 'DOCUMENT_UPLOADED'`, env.Data.ID).Scan(&payload); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(payload, "%PDF") || strings.Contains(payload, "minioadmin") || strings.Contains(strings.ToLower(payload), "secret") {
		t.Fatalf("audit leaked: %s", payload)
	}
	if !strings.Contains(payload, env.Data.Checksum) {
		t.Fatalf("audit missing checksum: %s", payload)
	}
	again := upload(t, eng, token, pdf, fields)
	if again.Code != http.StatusConflict {
		t.Fatalf("dedupe %d %s", again.Code, again.Body.String())
	}
	if objects.Len() != 1 {
		t.Fatalf("objects=%d", objects.Len())
	}

	_, err := pool.Exec(ctx, `INSERT INTO communes (id, name) VALUES ('xa_other', 'Other') ON CONFLICT (id) DO NOTHING`)
	if err != nil {
		t.Fatal(err)
	}
	otherID := uuid.New()
	_, err = pool.Exec(ctx, `
		INSERT INTO documents (
			id, xa_id, domain_id, title, filename, storage_uri, checksum, mime_type,
			file_size_bytes, processing_status, validity_status, uploaded_by
		) VALUES (
			$1, 'xa_other', 'ho_tich_chung_thuc', 'secret-other', 'o.pdf', 's3://cas-documents/hidden',
			repeat('ab', 32), 'application/pdf', 10, 'UPLOADED', 'PENDING',
			'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'
		)`, otherID)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_, _ = pool.Exec(context.Background(), `DELETE FROM documents WHERE id = $1`, otherID)
		_, _ = pool.Exec(context.Background(), `DELETE FROM communes WHERE id = 'xa_other'`)
	})
	req := httptest.NewRequest(http.MethodGet, "/api/v1/admin/documents/"+otherID.String(), nil)
	req.Header.Set("Authorization", "Bearer "+token)
	w := httptest.NewRecorder()
	eng.ServeHTTP(w, req)
	if w.Code != http.StatusNotFound || strings.Contains(w.Body.String(), "secret-other") {
		t.Fatalf("cross commune %d %s", w.Code, w.Body.String())
	}
}

func TestIntegrationConcurrentDuplicate(t *testing.T) {
	pool := requireDB(t)
	objects := storage.NewMemory()
	eng, token := engine(t, pool, objects)
	pdf := append([]byte("%PDF-1.4\n"), []byte("concurrent-unique")...)
	fields := map[string]string{"title": "Concurrent", "domain_id": "ho_tich_chung_thuc"}
	var wg sync.WaitGroup
	codes := make(chan int, 2)
	bodies := make(chan string, 2)
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			res := upload(t, eng, token, pdf, fields)
			codes <- res.Code
			bodies <- res.Body.String()
		}()
	}
	wg.Wait()
	close(codes)
	close(bodies)
	var created, conflict int
	var raw []string
	for code := range codes {
		switch code {
		case http.StatusCreated:
			created++
		case http.StatusConflict:
			conflict++
		}
	}
	for b := range bodies {
		raw = append(raw, b)
	}
	if created != 1 || conflict != 1 || objects.Len() != 1 {
		t.Fatalf("created=%d conflict=%d objects=%d bodies=%v", created, conflict, objects.Len(), raw)
	}
	var id string
	_ = pool.QueryRow(context.Background(), `SELECT id::text FROM documents WHERE title = 'Concurrent'`).Scan(&id)
	if id != "" {
		t.Cleanup(func() {
			_, _ = pool.Exec(context.Background(), `DELETE FROM audit_logs WHERE entity_id = $1`, id)
			_, _ = pool.Exec(context.Background(), `DELETE FROM documents WHERE id = $1`, id)
		})
	}
}
