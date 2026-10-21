package httpserver_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"mime/multipart"
	"net"
	"net/http"
	"net/http/httptest"
	"net/textproto"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/auth"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/config"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/httpserver"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/index"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/repository"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/storage"
)

const (
	p4bDSN           = "postgres://cas:cas@127.0.0.1:5434/citizen_assistance_p4b_test?sslmode=disable"
	p4bMinioEndpoint = "127.0.0.1:9010"
	p4bMinioBucket   = "cas-p4b-integration-test"
	p4bQdrantURL     = "http://127.0.0.1:6335"
	p4bWorkerToken   = "p4b-lifecycle-token"
)

func TestP4BRealMinIOWorkerActivateChatHTTP(t *testing.T) {
	pool := p4bWorkerPool(t)
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()

	pdfPath := filepath.Join(repoRoot(t), "apps", "ai-service", "tests", "fixtures", "native.pdf")
	pdf, err := os.ReadFile(pdfPath)
	if err != nil {
		t.Fatal(err)
	}
	pdf = append(append([]byte{}, pdf...), []byte("\n%"+uuid.NewString()+"\n")...)
	objects, err := storage.NewMinIO(storage.MinIOConfig{
		Endpoint: p4bMinioEndpoint, AccessKey: "minioadmin", SecretKey: "minioadmin",
		Bucket: p4bMinioBucket, UseSSL: false, AutoCreateBucket: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := objects.EnsureBucket(ctx); err != nil {
		t.Fatal(err)
	}

	workerURL := startP4BWorker(t)
	intentPhrase := "zxqvmnp4_" + strings.ReplaceAll(uuid.NewString(), "-", "")[:12]
	draftMock := mockDraftServer(t, intentPhrase)
	admin := uuid.MustParse("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
	cfg := config.Config{
		Env: "local", XAID: "xa_chu_se", LogLevel: "error",
		CitizenDomainIDs: []string{"ho_tich_chung_thuc"},
		JWTSecret:        "p45-lifecycle-jwt-secret-key", JWTExpireHours: 24,
		AdminIngestionEnabled: true, AdminIndexingEnabled: true, IndexMode: "pipeline",
		ObjectStorageBucket: p4bMinioBucket, ObjectStorageSSLConfigured: true,
		DocumentMaxBytes: 5 << 20, IndexClaimTTL: 3 * time.Minute,
		IndexPipelineTimeout: 90 * time.Second, IndexPublishTimeout: 20 * time.Second,
		AIServiceURL: draftMock.URL, AIServiceToken: p4bWorkerToken, AIDraftTimeoutMS: 20000,
	}
	worker := index.HTTPWorker{BaseURL: workerURL, Token: p4bWorkerToken, Timeout: 90 * time.Second}
	eng := httpserver.NewWithIndexWorker(slog.New(slog.NewTextHandler(io.Discard, nil)), pool, cfg, objects, worker)
	tokens := &auth.TokenService{Secret: []byte(cfg.JWTSecret), TTL: time.Hour}
	adminTok, _, err := tokens.Issue(admin, auth.RoleAdmin, "admin@chuse.vn", "Admin")
	if err != nil {
		t.Fatal(err)
	}

	code := "w2e_" + strings.ReplaceAll(uuid.NewString(), "-", "")[:10]
	var procID uuid.UUID
	if err := pool.QueryRow(ctx, `
		INSERT INTO procedures (procedure_code, domain_id, name, xa_id)
		VALUES ($1,'ho_tich_chung_thuc','Demo lifecycle P45','xa_chu_se')
		RETURNING id`, code).Scan(&procID); err != nil {
		t.Fatal(err)
	}
	defV0 := demoDefinition(code, "DEMO P45 V0", intentPhrase)
	v0 := apiJSON(t, eng, http.MethodPost, "/api/v1/admin/procedures/"+procID.String()+"/versions",
		map[string]any{"confirm": true, "version": "1.0.0", "definition": json.RawMessage(defV0)},
		adminHeaders(adminTok, uuid.NewString()))
	if v0.code != http.StatusCreated {
		t.Fatalf("create V0 %d %s", v0.code, v0.raw)
	}
	v0ID := uuid.MustParse(asString(v0.data["id"]))

	docID := uploadPDF(t, eng, adminTok, pdf)
	link := apiJSON(t, eng, http.MethodPost, "/api/v1/admin/documents/"+docID+"/links",
		map[string]any{"procedure_version_id": v0ID.String(), "relationship_type": "SOURCE", "page_range": "1-1"},
		adminHeaders(adminTok, uuid.NewString()))
	if link.code != http.StatusCreated {
		t.Fatalf("link V0 %d %s", link.code, link.raw)
	}
	idx0 := apiJSON(t, eng, http.MethodPost, "/api/v1/admin/documents/"+docID+"/links/"+v0ID.String()+"/index",
		nil, adminHeaders(adminTok, uuid.NewString()))
	if idx0.code != http.StatusOK || asString(idx0.data["link_status"]) != "READY" {
		t.Fatalf("index V0 %d %s", idx0.code, idx0.raw)
	}
	assertRealReady(t, ctx, pool, uuid.MustParse(docID), v0ID)

	act0 := apiJSON(t, eng, http.MethodPost, "/api/v1/admin/procedures/"+procID.String()+"/versions/"+v0ID.String()+"/activate",
		map[string]any{"expected_active_version_id": nil}, adminHeaders(adminTok, uuid.NewString()))
	if act0.code != http.StatusOK {
		t.Fatalf("activate V0 %d %s", act0.code, act0.raw)
	}

	oldSess, oldGuest := createSession(t, eng)
	oldTurn := apiJSON(t, eng, http.MethodPost, "/api/v1/sessions/"+oldSess+"/turns",
		map[string]any{"message": intentPhrase},
		map[string]string{"X-Guest-Token": oldGuest, "X-Request-ID": uuid.NewString()})
	if oldTurn.code != http.StatusOK {
		t.Fatalf("old chat %d %s", oldTurn.code, oldTurn.raw)
	}
	if asString(oldTurn.data["procedure_version"]) != "1.0.0" {
		t.Fatalf("old session pin %s", oldTurn.raw)
	}

	gen := apiJSON(t, eng, http.MethodPost, "/api/v1/admin/procedures/"+procID.String()+"/ai-drafts",
		map[string]any{"document_id": docID, "source_page_start": 1, "source_page_end": 1},
		adminHeaders(adminTok, uuid.NewString()))
	if gen.code != http.StatusCreated {
		t.Fatalf("generate %d %s", gen.code, gen.raw)
	}
	draftID := asString(gen.data["id"])
	got := apiJSON(t, eng, http.MethodGet, "/api/v1/admin/procedure-drafts/"+draftID, nil, adminHeaders(adminTok, ""))
	updated := asString(got.data["updated_at"])
	var decisions []map[string]string
	if props, ok := got.data["proposals"].([]any); ok {
		for _, rawProp := range props {
			p, _ := rawProp.(map[string]any)
			decisions = append(decisions, map[string]string{"proposal_id": asString(p["id"]), "decision": "accepted"})
		}
	}
	rev := apiJSON(t, eng, http.MethodPost, "/api/v1/admin/procedure-drafts/"+draftID+"/review",
		map[string]any{"expected_updated_at": updated, "decisions": decisions},
		adminHeaders(adminTok, uuid.NewString()))
	if rev.code != http.StatusOK {
		t.Fatalf("review %d %s", rev.code, rev.raw)
	}
	updated = asString(rev.data["updated_at"])
	appr := apiJSON(t, eng, http.MethodPost, "/api/v1/admin/procedure-drafts/"+draftID+"/approve",
		map[string]any{"confirm": true, "expected_updated_at": updated},
		adminHeaders(adminTok, uuid.NewString()))
	if appr.code != http.StatusOK || asString(appr.data["status"]) != repository.DraftStatusApproved {
		t.Fatalf("approve %d %s", appr.code, appr.raw)
	}
	updated = asString(appr.data["updated_at"])

	actDraft := apiJSON(t, eng, http.MethodPost, "/api/v1/admin/procedure-drafts/"+draftID+"/activate",
		map[string]any{"confirm": true, "expected_updated_at": updated, "page_range": "1-1", "expected_active_version_id": v0ID.String()},
		adminHeaders(adminTok, uuid.NewString()))
	if actDraft.code != http.StatusOK {
		t.Fatalf("activate draft %d %s", actDraft.code, actDraft.raw)
	}
	if actDraft.data["activated"] != true {
		t.Fatalf("expected ACTIVE after real worker, got %s", actDraft.raw)
	}
	ver, _ := actDraft.data["version"].(map[string]any)
	v1ID := uuid.MustParse(asString(ver["id"]))
	v1Label := asString(ver["version"])
	if asString(actDraft.data["draft"].(map[string]any)["status"]) != repository.DraftStatusPublished {
		t.Fatalf("draft not PUBLISHED: %s", actDraft.raw)
	}
	assertRealReady(t, ctx, pool, uuid.MustParse(docID), v1ID)

	retry := apiJSON(t, eng, http.MethodPost, "/api/v1/admin/procedure-drafts/"+draftID+"/activate",
		map[string]any{"confirm": true}, adminHeaders(adminTok, uuid.NewString()))
	if retry.code != http.StatusOK {
		t.Fatalf("retry %d %s", retry.code, retry.raw)
	}
	retryVer, _ := retry.data["version"].(map[string]any)
	if asString(retryVer["id"]) != v1ID.String() {
		t.Fatalf("retry created another version %s", retry.raw)
	}

	newSess, newGuest := createSession(t, eng)
	newTurn := apiJSON(t, eng, http.MethodPost, "/api/v1/sessions/"+newSess+"/turns",
		map[string]any{"message": intentPhrase},
		map[string]string{"X-Guest-Token": newGuest, "X-Request-ID": uuid.NewString()})
	if newTurn.code != http.StatusOK {
		t.Fatalf("new chat %d %s", newTurn.code, newTurn.raw)
	}
	if asString(newTurn.data["procedure_version"]) != v1Label {
		t.Fatalf("new session wanted %s got %s", v1Label, newTurn.raw)
	}
	if !strings.Contains(asString(newTurn.data["reply_text"])+asString(newTurn.data["guidance"]), "DEMO P45 V1") &&
		!strings.Contains(string(mustJSON(newTurn.data)), "DEMO P45 V1") {
		t.Fatalf("new chat did not follow V1 definition: %s", newTurn.raw)
	}

	hist := apiJSON(t, eng, http.MethodPost, "/api/v1/sessions/"+oldSess+"/turns",
		map[string]any{"message": "Nơi sinh Pleiku"},
		map[string]string{"X-Guest-Token": oldGuest, "X-Request-ID": uuid.NewString()})
	if hist.code != http.StatusOK {
		t.Fatalf("historical chat %d %s", hist.code, hist.raw)
	}
	var pinned uuid.UUID
	if err := pool.QueryRow(ctx, `SELECT active_procedure_version_id FROM conversation_sessions WHERE id=$1`, oldSess).Scan(&pinned); err != nil {
		t.Fatal(err)
	}
	if pinned != v0ID {
		t.Fatalf("old pin drifted to %s want %s", pinned, v0ID)
	}

	t.Logf("lifecycle PASS procedure=%s v0=%s v1=%s draft=%s doc=%s", code, v0ID, v1ID, draftID, docID)
}

func p4bWorkerPool(t *testing.T) *pgxpool.Pool {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	pool, err := pgxpool.New(ctx, p4bDSN)
	if err != nil {
		t.Skip("p4b-test postgres not reachable")
	}
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		t.Skip("p4b-test postgres not reachable")
	}
	var name string
	if err := pool.QueryRow(ctx, `SELECT current_database()`).Scan(&name); err != nil || name != "citizen_assistance_p4b_test" {
		pool.Close()
		t.Fatal("refusing non-p4b-test database")
	}
	t.Cleanup(pool.Close)
	return pool
}

func repoRoot(t *testing.T) string {
	t.Helper()
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("caller")
	}
	dir := filepath.Dir(file)
	for i := 0; i < 12; i++ {
		if _, err := os.Stat(filepath.Join(dir, "apps", "ai-service", "tests", "fixtures", "native.pdf")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	t.Fatal("repo root not found")
	return ""
}

type p4bWorkerIdentity struct {
	PostgresHost     string `json:"postgres_host"`
	PostgresPort     string `json:"postgres_port"`
	Database         string `json:"database"`
	MinioEndpoint    string `json:"minio_endpoint"`
	MinioBucket      string `json:"minio_bucket"`
	QdrantURL        string `json:"qdrant_url"`
	QdrantCollection string `json:"qdrant_collection"`
	IndexMode        string `json:"index_mode"`
	Embedding        string `json:"embedding_provider"`
	OCR              string `json:"ocr_provider"`
}

func expectedP4BWorkerIdentity() p4bWorkerIdentity {
	return p4bWorkerIdentity{
		PostgresHost: "127.0.0.1", PostgresPort: "5434", Database: "citizen_assistance_p4b_test",
		MinioEndpoint: p4bMinioEndpoint, MinioBucket: p4bMinioBucket,
		QdrantURL: p4bQdrantURL, QdrantCollection: "p4b_e5",
		IndexMode: "pipeline", Embedding: "fake", OCR: "fake",
	}
}

func (id p4bWorkerIdentity) Validate() error {
	if id.PostgresHost == "" || id.Database == "" || id.MinioEndpoint == "" || id.QdrantURL == "" {
		return fmt.Errorf("worker identity cannot be inspected")
	}
	if id.PostgresPort == "5432" || id.Database == "citizen_assistance" {
		return fmt.Errorf("worker points to shared postgresql")
	}
	if strings.Contains(id.MinioEndpoint, ":9000") || id.MinioBucket == "cas-documents" {
		return fmt.Errorf("worker points to shared minio")
	}
	if strings.Contains(id.QdrantURL, ":6333") {
		return fmt.Errorf("worker points to shared qdrant")
	}
	exp := expectedP4BWorkerIdentity()
	if id.PostgresHost != exp.PostgresHost || id.PostgresPort != exp.PostgresPort || id.Database != exp.Database ||
		id.MinioEndpoint != exp.MinioEndpoint || id.MinioBucket != exp.MinioBucket ||
		id.QdrantURL != exp.QdrantURL || id.QdrantCollection != exp.QdrantCollection ||
		id.IndexMode != exp.IndexMode || id.Embedding != exp.Embedding || id.OCR != exp.OCR {
		return fmt.Errorf("worker belongs to the wrong environment")
	}
	return nil
}

func verifyP4BWorker(baseURL string, id p4bWorkerIdentity) error {
	if strings.TrimSpace(baseURL) == "" {
		return fmt.Errorf("worker endpoint unavailable")
	}
	if err := id.Validate(); err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 800*time.Millisecond)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(baseURL, "/")+"/ready", nil)
	if err != nil {
		return fmt.Errorf("worker endpoint unavailable: %w", err)
	}
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("worker endpoint unavailable: %w", err)
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		return fmt.Errorf("worker endpoint unavailable: status %d", res.StatusCode)
	}
	var body struct {
		IndexMode         string `json:"index_mode"`
		EmbeddingProvider string `json:"embedding_provider"`
		OCRProvider       string `json:"ocr_provider"`
	}
	if err := json.NewDecoder(res.Body).Decode(&body); err != nil {
		return fmt.Errorf("worker identity cannot be inspected: %w", err)
	}
	if body.IndexMode != id.IndexMode || body.EmbeddingProvider != id.Embedding || body.OCRProvider != id.OCR {
		return fmt.Errorf("worker belongs to the wrong environment")
	}
	return nil
}

func rejectDeclaredP4BWorkerURL(rawURL string) error {
	if strings.TrimSpace(rawURL) != "" {
		return fmt.Errorf("CAS_P4B_WORKER_URL reuse is not allowed; p4b-test always starts a dedicated worker")
	}
	return nil
}

func startP4BWorker(t *testing.T) string {
	t.Helper()
	id := expectedP4BWorkerIdentity()
	if err := rejectDeclaredP4BWorkerURL(os.Getenv("CAS_P4B_WORKER_URL")); err != nil {
		t.Fatal(err)
	}
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := ln.Addr().(*net.TCPAddr).Port
	_ = ln.Close()
	baseURL := fmt.Sprintf("http://127.0.0.1:%d", port)
	aiDir := filepath.Join(repoRoot(t), "apps", "ai-service")
	py := filepath.Join(aiDir, ".venv", "bin", "python")
	cmd := exec.Command(py, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", fmt.Sprintf("%d", port))
	cmd.Dir = aiDir
	cmd.Env = []string{
		"PATH=" + filepath.Dir(py) + ":" + os.Getenv("PATH"),
		"HOME=" + os.Getenv("HOME"),
		"VIRTUAL_ENV=" + filepath.Dir(filepath.Dir(py)),
		"INDEX_MODE=" + id.IndexMode,
		"EMBEDDING_PROVIDER=" + id.Embedding,
		"OCR_PROVIDER=" + id.OCR,
		"DATABASE_URL=" + p4bDSN,
		"QDRANT_URL=" + id.QdrantURL,
		"QDRANT_COLLECTION=" + id.QdrantCollection,
		"OBJECT_STORAGE_ENDPOINT=" + id.MinioEndpoint,
		"OBJECT_STORAGE_ACCESS_KEY=minioadmin",
		"OBJECT_STORAGE_SECRET_KEY=minioadmin",
		"OBJECT_STORAGE_BUCKET=" + id.MinioBucket,
		"OBJECT_STORAGE_USE_SSL=false",
		"OBJECT_STORAGE_AUTO_CREATE_BUCKET=true",
		"AI_SERVICE_TOKEN=" + p4bWorkerToken,
		"INDEX_PIPELINE_TIMEOUT_SECONDS=90",
		"PYTHONPATH=" + aiDir,
	}
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		t.Fatalf("start worker: %v", err)
	}
	t.Cleanup(func() { _ = cmd.Process.Kill() })
	deadline := time.Now().Add(25 * time.Second)
	for time.Now().Before(deadline) {
		if err := verifyP4BWorker(baseURL, id); err == nil {
			return baseURL
		}
		time.Sleep(300 * time.Millisecond)
	}
	t.Fatal("dedicated p4b-test python worker did not become ready")
	return ""
}

func mockDraftServer(t *testing.T, intentPhrase string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer "+p4bWorkerToken {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		var req struct {
			RequestID     string `json:"request_id"`
			ProcedureCode string `json:"procedure_code"`
			DocumentID    string `json:"document_id"`
			Excerpts      []struct {
				ExcerptID string `json:"excerpt_id"`
				Content   string `json:"content"`
				PageStart int    `json:"page_start"`
			} `json:"excerpts"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		quote := "uy ban nhan dan xa Chu Se"
		if len(req.Excerpts) > 0 {
			if !strings.Contains(req.Excerpts[0].Content, quote) {
				quote = strings.TrimSpace(req.Excerpts[0].Content)
				if len(quote) > 80 {
					quote = quote[:80]
				}
			}
		}
		def := map[string]any{
			"procedure_code":  req.ProcedureCode,
			"name":            "Demo lifecycle P45",
			"intent_examples": []string{intentPhrase},
			"slots":           map[string]any{"noi_sinh": map[string]any{"type": "string", "question": "Nơi sinh?"}},
			"required_slots":  []string{"noi_sinh"},
			"guidance":        map[string]any{"summary": "Nộp tại UBND — DEMO P45 V1", "checklist": []string{}},
			"citations":       []map[string]any{{"doc_id": req.DocumentID, "title": "native.pdf", "source_type": "ai_draft_unapproved"}},
		}
		defRaw, _ := json.Marshal(def)
		exID := ""
		page := 1
		if len(req.Excerpts) > 0 {
			exID = req.Excerpts[0].ExcerptID
			page = req.Excerpts[0].PageStart
		}
		body, _ := json.Marshal(map[string]any{
			"schema_version": "draft_definition.v1", "request_id": req.RequestID,
			"procedure_code": req.ProcedureCode, "original_procedure_code": req.ProcedureCode,
			"definition": json.RawMessage(defRaw),
			"evidence": []map[string]any{
				{"proposal_id": "slot:noi_sinh", "kind": "slot", "excerpt_id": exID, "page": page, "quotation": quote},
				{"proposal_id": "required:noi_sinh", "kind": "required_slot", "excerpt_id": exID, "page": page, "quotation": quote},
				{"proposal_id": "guidance:summary", "kind": "guidance", "excerpt_id": exID, "page": page, "quotation": quote},
			},
			"warnings": []string{}, "missing_requirements": []string{},
			"validation": map[string]any{"schema_valid": true, "grounding_verified": false, "semantic_approved": false, "issues": []any{}, "readiness": "needs_review"},
			"provider":   "demo-test", "model": "fixture-not-gemini", "prompt_version": "draft.prompt.v2", "prompt_hash": "demo",
			"usage": map[string]any{"total_token_count": 1},
		})
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(body)
	}))
	t.Cleanup(srv.Close)
	return srv
}

func demoDefinition(code, marker, intentPhrase string) json.RawMessage {
	raw, _ := json.Marshal(map[string]any{
		"procedure_code":  code,
		"name":            "Demo lifecycle P45",
		"intent_examples": []string{intentPhrase},
		"slots":           map[string]any{"noi_sinh": map[string]any{"type": "string", "question": "Nơi sinh?"}},
		"required_slots":  []string{"noi_sinh"},
		"guidance":        map[string]any{"summary": "Nộp tại UBND — " + marker, "checklist": []string{"Giấy tờ tùy thân"}},
	})
	return raw
}

func assertRealReady(t *testing.T, ctx context.Context, pool *pgxpool.Pool, docID, versionID uuid.UUID) {
	t.Helper()
	var status, pipeline string
	var chunks, vectors int
	err := pool.QueryRow(ctx, `
		SELECT g.status, g.pipeline_version, g.chunk_count, g.vector_count
		FROM document_index_generations g
		JOIN procedure_version_documents l ON l.active_generation_id = g.id
		WHERE l.document_id=$1 AND l.procedure_version_id=$2`, docID, versionID).
		Scan(&status, &pipeline, &chunks, &vectors)
	if err != nil {
		t.Fatal(err)
	}
	if status != "READY" || chunks < 1 || vectors != chunks {
		t.Fatalf("generation not really READY status=%s chunks=%d vectors=%d pipeline=%s", status, chunks, vectors, pipeline)
	}
	if pipeline == "p4-test" {
		t.Fatal("fixture markSourceReady pipeline_version leaked into real worker E2E")
	}
}

func uploadPDF(t *testing.T, eng http.Handler, token string, pdf []byte) string {
	t.Helper()
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)
	_ = w.WriteField("title", "DEMO P45 native")
	_ = w.WriteField("domain_id", "ho_tich_chung_thuc")
	hdr := make(textproto.MIMEHeader)
	hdr.Set("Content-Disposition", `form-data; name="file"; filename="native.pdf"`)
	hdr.Set("Content-Type", "application/pdf")
	part, err := w.CreatePart(hdr)
	if err != nil {
		t.Fatal(err)
	}
	_, _ = part.Write(pdf)
	_ = w.Close()
	req := httptest.NewRequest(http.MethodPost, "/api/v1/admin/documents", &buf)
	req.Header.Set("Content-Type", w.FormDataContentType())
	req.Header.Set("Authorization", "Bearer "+token)
	res := httptest.NewRecorder()
	eng.ServeHTTP(res, req)
	if res.Code != http.StatusCreated {
		t.Fatalf("upload %d %s", res.Code, res.Body.String())
	}
	var env struct {
		Data struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.Unmarshal(res.Body.Bytes(), &env); err != nil {
		t.Fatal(err)
	}
	return env.Data.ID
}

func createSession(t *testing.T, eng http.Handler) (string, string) {
	t.Helper()
	res := apiJSON(t, eng, http.MethodPost, "/api/v1/sessions", map[string]any{"xa_id": "xa_chu_se"}, nil)
	if res.code != http.StatusCreated {
		t.Fatalf("session %d %s", res.code, res.raw)
	}
	return asString(res.data["id"]), asString(res.data["guest_token"])
}

func adminHeaders(token, requestID string) map[string]string {
	h := map[string]string{"Authorization": "Bearer " + token}
	if requestID != "" {
		h["X-Request-ID"] = requestID
	}
	return h
}

type httpEnv struct {
	code int
	raw  string
	data map[string]any
}

func apiJSON(t *testing.T, eng http.Handler, method, path string, body any, headers map[string]string) httpEnv {
	t.Helper()
	var buf bytes.Buffer
	if body != nil {
		if err := json.NewEncoder(&buf).Encode(body); err != nil {
			t.Fatal(err)
		}
	}
	req := httptest.NewRequest(method, path, &buf)
	req.Header.Set("Content-Type", "application/json")
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	w := httptest.NewRecorder()
	eng.ServeHTTP(w, req)
	var env map[string]any
	_ = json.Unmarshal(w.Body.Bytes(), &env)
	data, _ := env["data"].(map[string]any)
	if data == nil {
		data = env
	}
	return httpEnv{code: w.Code, raw: w.Body.String(), data: data}
}

func asString(v any) string {
	switch x := v.(type) {
	case string:
		return x
	case map[string]any:
		if s, ok := x["status"].(string); ok {
			return s
		}
	}
	if v == nil {
		return ""
	}
	b, _ := json.Marshal(v)
	return strings.Trim(string(b), `"`)
}

func mustJSON(v any) []byte {
	b, _ := json.Marshal(v)
	return b
}

func contextWithTimeout() (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.Background(), 20*time.Second)
}

func mustAdmin() uuid.UUID {
	return uuid.MustParse("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
}

func shortID() string {
	return strings.ReplaceAll(uuid.NewString(), "-", "")[:8]
}

func archiveFixtureOnly(ctx context.Context, pool *pgxpool.Pool, procedureID string) {
	_, _ = pool.Exec(ctx, `UPDATE procedure_versions SET status='ARCHIVED' WHERE procedure_id=$1::uuid AND status='ACTIVE'`, procedureID)
	_, _ = pool.Exec(ctx, `UPDATE procedures SET active_version_id=NULL WHERE id=$1::uuid`, procedureID)
}
