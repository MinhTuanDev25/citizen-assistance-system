//go:build integration

package index_test

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
	"sync/atomic"
	"testing"
	"time"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/auth"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/config"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/httpserver"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/index"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/storage"
	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestTwoLinksDoNotShareReadyAndRecompute(t *testing.T) {
	pool := requireDB(t)
	eng, admin := engine(t, pool, readyWorker(), true)
	docID := uploadDoc(t, eng, admin, "%PDF-1.4 two-links")
	t.Cleanup(func() { cleanupDoc(pool, docID) })
	first := oneVersion(t, pool, "xa_chu_se", "ho_tich_chung_thuc")
	second := insertVersion(t, pool, "xa_chu_se", "ho_tich_chung_thuc", "p4a_b_"+uuid.NewString()[:8])
	t.Cleanup(func() { cleanupVersion(pool, second) })
	linkDoc(t, eng, admin, docID, first)
	linkDoc(t, eng, admin, docID, second)
	res := call(t, eng, http.MethodPost, indexPath(docID, first, false), admin, uuid.New(), nil)
	if res.Code != http.StatusOK || !bytes.Contains(res.Body.Bytes(), []byte(`"link_status":"READY"`)) {
		t.Fatalf("index first %d %s", res.Code, res.Body.String())
	}
	if got, _ := statuses(t, pool, docID, second); got != "UPLOADED" {
		t.Fatalf("second link became %s", got)
	}
	if _, doc := statuses(t, pool, docID, first); doc == "READY" {
		t.Fatal("document aggregate became READY while a link is still UPLOADED")
	}
	secondRes := call(t, eng, http.MethodPost, indexPath(docID, second, false), admin, uuid.New(), nil)
	if secondRes.Code != http.StatusOK {
		t.Fatalf("index second %d %s", secondRes.Code, secondRes.Body.String())
	}
	if _, doc := statuses(t, pool, docID, first); doc != "READY" {
		t.Fatalf("aggregate %s", doc)
	}
	third := insertVersion(t, pool, "xa_chu_se", "ho_tich_chung_thuc", "p4a_c_"+uuid.NewString()[:8])
	t.Cleanup(func() { cleanupVersion(pool, third) })
	linkDoc(t, eng, admin, docID, third)
	if _, doc := statuses(t, pool, docID, third); doc == "READY" {
		t.Fatal("new link left the document READY")
	}
	unlink(t, eng, admin, docID, third)
	if _, doc := statuses(t, pool, docID, first); doc != "READY" {
		t.Fatalf("aggregate after unlink %s", doc)
	}
	unlink(t, eng, admin, docID, first)
	unlink(t, eng, admin, docID, second)
	if _, doc := statuses(t, pool, docID, first); doc != "UPLOADED" {
		t.Fatalf("aggregate with no links %s", doc)
	}
	assertChunks(t, pool, docID, 0)
}

func TestSameRequestIDRunsWorkerOnce(t *testing.T) {
	pool := requireDB(t)
	versionID := oneVersion(t, pool, "xa_chu_se", "ho_tich_chung_thuc")
	var calls atomic.Int32
	started := make(chan struct{})
	release := make(chan struct{})
	worker := workerFn(func(_ context.Context, req index.Request) (index.Response, error) {
		if calls.Add(1) == 1 {
			close(started)
			<-release
		}
		return readyResponse(req), nil
	})
	eng, admin := engine(t, pool, worker, true)
	docID := uploadDoc(t, eng, admin, "%PDF-1.4 same-request")
	t.Cleanup(func() { cleanupDoc(pool, docID) })
	linkDoc(t, eng, admin, docID, versionID)
	reqID := uuid.New()
	var wg sync.WaitGroup
	codes := make(chan string, 2)
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			res := call(t, eng, http.MethodPost, indexPath(docID, versionID, false), admin, reqID, nil)
			codes <- res.Body.String()
		}()
	}
	select {
	case <-started:
	case <-time.After(5 * time.Second):
		t.Fatal("worker did not start")
	}
	close(release)
	wg.Wait()
	close(codes)
	for body := range codes {
		if !strings.Contains(body, `"link_status"`) {
			t.Fatalf("body %s", body)
		}
	}
	if calls.Load() != 1 {
		t.Fatalf("worker calls %d", calls.Load())
	}
}

func TestDifferentRequestIDKeepsOneClaim(t *testing.T) {
	pool := requireDB(t)
	versionID := oneVersion(t, pool, "xa_chu_se", "ho_tich_chung_thuc")
	var calls atomic.Int32
	started := make(chan struct{})
	release := make(chan struct{})
	worker := workerFn(func(_ context.Context, req index.Request) (index.Response, error) {
		if calls.Add(1) == 1 {
			close(started)
			<-release
		}
		return readyResponse(req), nil
	})
	eng, admin := engine(t, pool, worker, true)
	docID := uploadDoc(t, eng, admin, "%PDF-1.4 two-request")
	t.Cleanup(func() { cleanupDoc(pool, docID) })
	linkDoc(t, eng, admin, docID, versionID)
	var wg sync.WaitGroup
	codes := make(chan int, 2)
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			codes <- call(t, eng, http.MethodPost, indexPath(docID, versionID, false), admin, uuid.New(), nil).Code
		}()
	}
	select {
	case <-started:
	case <-time.After(5 * time.Second):
		t.Fatal("worker did not start")
	}
	close(release)
	wg.Wait()
	close(codes)
	ok, rejected := 0, 0
	for code := range codes {
		if code == http.StatusOK {
			ok++
		} else if code == http.StatusConflict {
			rejected++
		} else {
			t.Fatalf("code %d", code)
		}
	}
	if ok != 1 || rejected != 1 || calls.Load() != 1 {
		t.Fatalf("ok=%d rejected=%d calls=%d", ok, rejected, calls.Load())
	}
	var jobs int
	if err := pool.QueryRow(context.Background(), `SELECT count(*) FROM document_index_jobs WHERE document_id = $1 AND status = 'CLAIMED'`, docID).Scan(&jobs); err != nil {
		t.Fatal(err)
	}
	if jobs != 0 {
		t.Fatalf("live claims %d", jobs)
	}
}

func TestExpiredLeaseLateWorkerDoesNotOverwrite(t *testing.T) {
	pool := requireDB(t)
	versionID := oneVersion(t, pool, "xa_chu_se", "ho_tich_chung_thuc")
	var calls atomic.Int32
	started := make(chan struct{})
	release := make(chan struct{})
	worker := workerFn(func(_ context.Context, req index.Request) (index.Response, error) {
		if calls.Add(1) == 1 {
			close(started)
			<-release
		}
		return readyResponse(req), nil
	})
	eng, admin := engine(t, pool, worker, true)
	docID := uploadDoc(t, eng, admin, "%PDF-1.4 lease")
	t.Cleanup(func() { cleanupDoc(pool, docID) })
	linkDoc(t, eng, admin, docID, versionID)
	lateCode := make(chan int, 1)
	go func() {
		lateCode <- call(t, eng, http.MethodPost, indexPath(docID, versionID, false), admin, uuid.New(), nil).Code
	}()
	select {
	case <-started:
	case <-time.After(5 * time.Second):
		t.Fatal("worker did not start")
	}
	if call(t, eng, http.MethodDelete, "/api/v1/admin/documents/"+docID+"/links/"+versionID, admin, uuid.New(), nil).Code != http.StatusConflict {
		t.Fatal("PROCESSING link was unlinked")
	}
	if _, err := pool.Exec(context.Background(), `
		UPDATE document_index_jobs
		SET claim_expires_at = now() - interval '1 second'
		WHERE document_id = $1 AND status = 'CLAIMED'`, docID); err != nil {
		t.Fatal(err)
	}
	recovered := call(t, eng, http.MethodPost, indexPath(docID, versionID, true), admin, uuid.New(), nil)
	if recovered.Code != http.StatusOK || !bytes.Contains(recovered.Body.Bytes(), []byte(`"link_status":"READY"`)) {
		t.Fatalf("recover %d %s", recovered.Code, recovered.Body.String())
	}
	close(release)
	if code := <-lateCode; code != http.StatusConflict {
		t.Fatalf("late worker HTTP %d", code)
	}
	link, doc := statuses(t, pool, docID, versionID)
	if link != "READY" || doc != "READY" {
		t.Fatalf("link %s doc %s", link, doc)
	}
	assertChunks(t, pool, docID, 0)
}

func TestCrossCommuneReplayStaysHidden(t *testing.T) {
	pool := requireDB(t)
	versionID := oneVersion(t, pool, "xa_chu_se", "ho_tich_chung_thuc")
	eng, admin := engine(t, pool, readyWorker(), true)
	docID := uploadDoc(t, eng, admin, "%PDF-1.4 isolation")
	t.Cleanup(func() { cleanupDoc(pool, docID) })
	linkDoc(t, eng, admin, docID, versionID)
	reqID := uuid.New()
	ok := call(t, eng, http.MethodPost, indexPath(docID, versionID, false), admin, reqID, nil)
	if ok.Code != http.StatusOK {
		t.Fatalf("index %d %s", ok.Code, ok.Body.String())
	}
	other := otherCommuneDoc(t, pool)
	t.Cleanup(func() { cleanupDoc(pool, other) })
	hidden := call(t, eng, http.MethodPost, indexPath(other, versionID, false), admin, reqID, nil)
	if hidden.Code != http.StatusNotFound {
		t.Fatalf("cross commune %d %s", hidden.Code, hidden.Body.String())
	}
	if bytes.Contains(hidden.Body.Bytes(), []byte(docID)) {
		t.Fatal("replay leaked the other commune document")
	}
	if call(t, eng, http.MethodGet, "/api/v1/admin/documents/link-targets", "", uuid.Nil, nil).Code != http.StatusUnauthorized {
		t.Fatal("missing token")
	}
	if call(t, eng, http.MethodGet, "/api/v1/admin/documents/link-targets", citizenToken(t), uuid.Nil, nil).Code != http.StatusForbidden {
		t.Fatal("citizen")
	}
	off, _ := engine(t, pool, nil, false)
	if call(t, off, http.MethodGet, "/api/v1/admin/documents/link-targets", admin, uuid.Nil, nil).Code != http.StatusNotFound {
		t.Fatal("flag off")
	}
}

func TestSameRequestDifferentPayloadConflicts(t *testing.T) {
	pool := requireDB(t)
	first := oneVersion(t, pool, "xa_chu_se", "ho_tich_chung_thuc")
	second := insertVersion(t, pool, "xa_chu_se", "ho_tich_chung_thuc", "p4a_d_"+uuid.NewString()[:8])
	t.Cleanup(func() { cleanupVersion(pool, second) })
	eng, admin := engine(t, pool, readyWorker(), true)
	docID := uploadDoc(t, eng, admin, "%PDF-1.4 payload")
	t.Cleanup(func() { cleanupDoc(pool, docID) })
	linkDoc(t, eng, admin, docID, first)
	linkDoc(t, eng, admin, docID, second)
	reqID := uuid.New()
	ok := call(t, eng, http.MethodPost, indexPath(docID, first, false), admin, reqID, nil)
	if ok.Code != http.StatusOK {
		t.Fatalf("index %d %s", ok.Code, ok.Body.String())
	}
	conflict := call(t, eng, http.MethodPost, indexPath(docID, second, false), admin, reqID, nil)
	if conflict.Code != http.StatusConflict || !bytes.Contains(conflict.Body.Bytes(), []byte("IDEMPOTENCY_CONFLICT")) {
		t.Fatalf("conflict %d %s", conflict.Code, conflict.Body.String())
	}
	if got, _ := statuses(t, pool, docID, second); got != "UPLOADED" {
		t.Fatalf("second link %s", got)
	}
}

func TestStrictBodyAndWorkerMismatch(t *testing.T) {
	pool := requireDB(t)
	versionID := oneVersion(t, pool, "xa_chu_se", "ho_tich_chung_thuc")
	eng, admin := engine(t, pool, workerFn(func(_ context.Context, req index.Request) (index.Response, error) {
		return index.Response{SchemaVersion: index.SchemaVersion, DocumentID: req.DocumentID, ProcedureVersionID: uuid.New(), Outcome: index.OutcomeReady}, nil
	}), true)
	docID := uploadDoc(t, eng, admin, "%PDF-1.4 strict")
	t.Cleanup(func() { cleanupDoc(pool, docID) })
	for _, raw := range [][]byte{
		[]byte(`{"procedure_version_id":"` + versionID + `","relationship_type":"SOURCE"}{"extra":true}`),
		[]byte(`{"procedure_version_id":"` + versionID + `","relationship_type":"SOURCE"} true`),
		[]byte(`{"procedure_version_id":"` + versionID + `","relationship_type":"SOURCE"}}`),
	} {
		req := httptest.NewRequest(http.MethodPost, "/api/v1/admin/documents/"+docID+"/links", bytes.NewReader(raw))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", "Bearer "+admin)
		req.Header.Set("X-Request-ID", uuid.NewString())
		res := httptest.NewRecorder()
		eng.ServeHTTP(res, req)
		if res.Code != http.StatusBadRequest {
			t.Fatalf("trailing %d %s", res.Code, res.Body.String())
		}
	}
	badRange := call(t, eng, http.MethodPost, "/api/v1/admin/documents/"+docID+"/links", admin, uuid.New(), map[string]string{
		"procedure_version_id": versionID, "relationship_type": "SOURCE", "page_range": "2-1",
	})
	if badRange.Code != http.StatusUnprocessableEntity {
		t.Fatalf("range %d %s", badRange.Code, badRange.Body.String())
	}
	linkDoc(t, eng, admin, docID, versionID)
	mismatched := call(t, eng, http.MethodPost, indexPath(docID, versionID, false), admin, uuid.New(), nil)
	if mismatched.Code != http.StatusOK || !bytes.Contains(mismatched.Body.Bytes(), []byte(`"error_code":"worker_mismatch"`)) {
		t.Fatalf("mismatch %d %s", mismatched.Code, mismatched.Body.String())
	}
	link, _ := statuses(t, pool, docID, versionID)
	if link != "FAILED" {
		t.Fatalf("link %s", link)
	}
	assertChunks(t, pool, docID, 0)
	if call(t, eng, http.MethodDelete, "/api/v1/admin/documents/"+docID+"/links/"+versionID, admin, uuid.New(), nil).Code == http.StatusConflict {
		t.Fatal("FAILED link refused unlink")
	}
}

func TestConcurrentLinkAndUnlinkReplayOnce(t *testing.T) {
	pool := requireDB(t)
	versionID := oneVersion(t, pool, "xa_chu_se", "ho_tich_chung_thuc")
	eng, admin := engine(t, pool, readyWorker(), true)
	docID := uploadDoc(t, eng, admin, "%PDF-1.4 concurrent-link")
	t.Cleanup(func() { cleanupDoc(pool, docID) })
	reqID := uuid.New()
	body := map[string]string{"procedure_version_id": versionID, "relationship_type": "SOURCE"}
	var wg sync.WaitGroup
	codes := make(chan int, 2)
	bodies := make(chan string, 2)
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			res := call(t, eng, http.MethodPost, "/api/v1/admin/documents/"+docID+"/links", admin, reqID, body)
			codes <- res.Code
			bodies <- res.Body.String()
		}()
	}
	wg.Wait()
	close(codes)
	close(bodies)
	created, replayed := 0, 0
	for code := range codes {
		switch code {
		case http.StatusCreated:
			created++
		case http.StatusOK:
			replayed++
		default:
			t.Fatalf("link code %d", code)
		}
	}
	if created != 1 || replayed != 1 {
		t.Fatalf("created=%d replayed=%d", created, replayed)
	}
	for raw := range bodies {
		if !strings.Contains(raw, versionID) {
			t.Fatalf("replay body %s", raw)
		}
	}
	var n int
	if err := pool.QueryRow(context.Background(), `SELECT count(*) FROM procedure_version_documents WHERE document_id = $1`, docID).Scan(&n); err != nil {
		t.Fatal(err)
	}
	if n != 1 {
		t.Fatalf("links %d", n)
	}
	conflict := call(t, eng, http.MethodPost, "/api/v1/admin/documents/"+docID+"/links", admin, reqID, map[string]string{
		"procedure_version_id": versionID, "relationship_type": "SOURCE", "page_range": "1-2",
	})
	if conflict.Code != http.StatusConflict || !bytes.Contains(conflict.Body.Bytes(), []byte("IDEMPOTENCY_CONFLICT")) {
		t.Fatalf("payload conflict %d %s", conflict.Code, conflict.Body.String())
	}
	unlinkID := uuid.New()
	codes = make(chan int, 2)
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			codes <- call(t, eng, http.MethodDelete, "/api/v1/admin/documents/"+docID+"/links/"+versionID, admin, unlinkID, nil).Code
		}()
	}
	wg.Wait()
	close(codes)
	for code := range codes {
		if code != http.StatusOK {
			t.Fatalf("unlink %d", code)
		}
	}
	if err := pool.QueryRow(context.Background(), `SELECT count(*) FROM procedure_version_documents WHERE document_id = $1 AND unlinked_at IS NULL`, docID).Scan(&n); err != nil {
		t.Fatal(err)
	}
	if n != 0 {
		t.Fatalf("active links after unlink %d", n)
	}
	if err := pool.QueryRow(context.Background(), `SELECT count(*) FROM procedure_version_documents WHERE document_id = $1 AND unlinked_at IS NOT NULL`, docID).Scan(&n); err != nil {
		t.Fatal(err)
	}
	if n != 1 {
		t.Fatalf("soft-unlinked rows %d", n)
	}
}

func TestLinkReplayDoesNotLeakAnotherCommune(t *testing.T) {
	pool := requireDB(t)
	versionID := oneVersion(t, pool, "xa_chu_se", "ho_tich_chung_thuc")
	eng, admin := engine(t, pool, readyWorker(), true)
	docID := uploadDoc(t, eng, admin, "%PDF-1.4 commune-link")
	t.Cleanup(func() { cleanupDoc(pool, docID) })
	reqID := uuid.New()
	ok := call(t, eng, http.MethodPost, "/api/v1/admin/documents/"+docID+"/links", admin, reqID, map[string]string{
		"procedure_version_id": versionID, "relationship_type": "SOURCE",
	})
	if ok.Code != http.StatusCreated {
		t.Fatalf("first link %d %s", ok.Code, ok.Body.String())
	}
	other := otherCommuneDoc(t, pool)
	t.Cleanup(func() { cleanupDoc(pool, other) })
	hidden := call(t, eng, http.MethodPost, "/api/v1/admin/documents/"+other+"/links", admin, reqID, map[string]string{
		"procedure_version_id": versionID, "relationship_type": "SOURCE",
	})
	if hidden.Code != http.StatusNotFound {
		t.Fatalf("cross commune %d %s", hidden.Code, hidden.Body.String())
	}
	if bytes.Contains(hidden.Body.Bytes(), []byte(docID)) {
		t.Fatal("replay leaked the other commune document")
	}
}

type workerFn func(context.Context, index.Request) (index.Response, error)

func (f workerFn) Index(ctx context.Context, req index.Request) (index.Response, error) {
	return f(ctx, req)
}

func readyWorker() index.Worker {
	return workerFn(func(_ context.Context, req index.Request) (index.Response, error) {
		return readyResponse(req), nil
	})
}

func readyResponse(req index.Request) index.Response {
	return index.Response{
		SchemaVersion: index.SchemaVersion, DocumentID: req.DocumentID,
		ProcedureVersionID: req.ProcedureVersionID, Outcome: index.OutcomeReady,
	}
}

func indexPath(docID, versionID string, retry bool) string {
	path := "/api/v1/admin/documents/" + docID + "/links/" + versionID + "/index"
	if retry {
		path += "/retry"
	}
	return path
}

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

func engine(t *testing.T, pool *pgxpool.Pool, worker index.Worker, indexing bool) (http.Handler, string) {
	t.Helper()
	cfg := config.Config{
		Env: "local", XAID: "xa_chu_se", LogLevel: "error",
		CitizenDomainIDs: []string{"ho_tich_chung_thuc"},
		JWTSecret:        "integration-test-jwt-secret-key", JWTExpireHours: 24,
		AdminIngestionEnabled: true, AdminIndexingEnabled: indexing,
		ObjectStorageBucket: "cas-documents", ObjectStorageSSLConfigured: true,
		DocumentMaxBytes: 1 << 20, IndexClaimTTL: 5 * time.Second, IndexTimeout: 8 * time.Second,
		AIServiceURL: "http://ai-service:8001", AIServiceToken: "integration-ai-token",
	}
	tokens := &auth.TokenService{Secret: []byte(cfg.JWTSecret), TTL: time.Hour}
	tok, _, err := tokens.Issue(uuid.MustParse("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"), auth.RoleAdmin, "admin@chuse.vn", "Seed Admin")
	if err != nil {
		t.Fatal(err)
	}
	return httpserver.NewWithIndexWorker(slog.New(slog.NewTextHandler(io.Discard, nil)), pool, cfg, storage.NewMemory(), worker), tok
}

func citizenToken(t *testing.T) string {
	t.Helper()
	tokens := &auth.TokenService{Secret: []byte("integration-test-jwt-secret-key"), TTL: time.Hour}
	tok, _, err := tokens.Issue(uuid.New(), auth.RoleCitizen, "c@example.com", "Citizen")
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

func uploadDoc(t *testing.T, eng http.Handler, token, body string) string {
	t.Helper()
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)
	_ = w.WriteField("title", "Tai lieu")
	_ = w.WriteField("domain_id", "ho_tich_chung_thuc")
	hdr := make(textproto.MIMEHeader)
	hdr.Set("Content-Disposition", `form-data; name="file"; filename="a.pdf"`)
	hdr.Set("Content-Type", "application/pdf")
	part, err := w.CreatePart(hdr)
	if err != nil {
		t.Fatal(err)
	}
	_, _ = part.Write([]byte(body))
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

func linkDoc(t *testing.T, eng http.Handler, token, docID, versionID string) {
	t.Helper()
	res := call(t, eng, http.MethodPost, "/api/v1/admin/documents/"+docID+"/links", token, uuid.New(), map[string]string{
		"procedure_version_id": versionID, "relationship_type": "SOURCE",
	})
	if res.Code != http.StatusCreated {
		t.Fatalf("link %d %s", res.Code, res.Body.String())
	}
}

func unlink(t *testing.T, eng http.Handler, token, docID, versionID string) {
	t.Helper()
	res := call(t, eng, http.MethodDelete, "/api/v1/admin/documents/"+docID+"/links/"+versionID, token, uuid.New(), nil)
	if res.Code != http.StatusOK {
		t.Fatalf("unlink %d %s", res.Code, res.Body.String())
	}
}

func call(t *testing.T, eng http.Handler, method, path, token string, requestID uuid.UUID, body map[string]string) *httptest.ResponseRecorder {
	t.Helper()
	var reader io.Reader
	if body != nil {
		raw, _ := json.Marshal(body)
		reader = bytes.NewReader(raw)
	}
	req := httptest.NewRequest(method, path, reader)
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	if requestID != uuid.Nil {
		req.Header.Set("X-Request-ID", requestID.String())
	}
	res := httptest.NewRecorder()
	eng.ServeHTTP(res, req)
	return res
}

func statuses(t *testing.T, pool *pgxpool.Pool, docID, versionID string) (string, string) {
	t.Helper()
	var link, doc string
	err := pool.QueryRow(context.Background(), `
		SELECT COALESCE((SELECT index_status FROM procedure_version_documents WHERE document_id = $1 AND procedure_version_id = $2), ''),
			processing_status
		FROM documents WHERE id = $1`, docID, versionID).Scan(&link, &doc)
	if err != nil {
		t.Fatal(err)
	}
	return link, doc
}

func oneVersion(t *testing.T, pool *pgxpool.Pool, xaID, domainID string) string {
	t.Helper()
	var id string
	err := pool.QueryRow(context.Background(), `
		SELECT v.id FROM procedure_versions v
		JOIN procedures p ON p.id = v.procedure_id
		WHERE p.xa_id = $1 AND p.domain_id = $2 AND v.status <> 'ARCHIVED'
		LIMIT 1`, xaID, domainID).Scan(&id)
	if err != nil {
		t.Fatal(err)
	}
	return id
}

func insertVersion(t *testing.T, pool *pgxpool.Pool, xaID, domainID, code string) string {
	t.Helper()
	procID := uuid.New()
	verID := uuid.New()
	admin := uuid.MustParse("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
	ctx := context.Background()
	if _, err := pool.Exec(ctx, `
		INSERT INTO procedures (id, procedure_code, domain_id, name, xa_id)
		VALUES ($1,$2,$3,$4,$5)`, procID, code, domainID, code, xaID); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO procedure_versions (id, procedure_id, version, status, definition, created_by)
		VALUES ($1,$2,'1.0.0','APPROVED','{}'::jsonb,$3)`, verID, procID, admin); err != nil {
		t.Fatal(err)
	}
	return verID.String()
}

func cleanupVersion(pool *pgxpool.Pool, versionID string) {
	ctx := context.Background()
	var procID string
	_ = pool.QueryRow(ctx, `SELECT procedure_id FROM procedure_versions WHERE id = $1`, versionID).Scan(&procID)
	_, _ = pool.Exec(ctx, `DELETE FROM document_index_jobs WHERE procedure_version_id = $1`, versionID)
	_, _ = pool.Exec(ctx, `DELETE FROM procedure_version_documents WHERE procedure_version_id = $1`, versionID)
	_, _ = pool.Exec(ctx, `DELETE FROM procedure_versions WHERE id = $1`, versionID)
	if procID != "" {
		_, _ = pool.Exec(ctx, `DELETE FROM procedures WHERE id = $1`, procID)
	}
}

func otherCommuneDoc(t *testing.T, pool *pgxpool.Pool) string {
	t.Helper()
	ctx := context.Background()
	_, _ = pool.Exec(ctx, `INSERT INTO communes (id, name) VALUES ('xa_p4a_other', 'Xa khac') ON CONFLICT (id) DO NOTHING`)
	id := uuid.New()
	admin := uuid.MustParse("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
	sum := strings.ReplaceAll(id.String(), "-", "") + strings.ReplaceAll(id.String(), "-", "")
	_, err := pool.Exec(ctx, `
		INSERT INTO documents (
			id, xa_id, domain_id, title, filename, storage_uri, checksum, mime_type, file_size_bytes,
			processing_status, validity_status, uploaded_by
		) VALUES (
			$1,'xa_p4a_other','ho_tich_chung_thuc','Ngoai','a.pdf','s3://cas-documents/x',$2,
			'application/pdf', 8, 'UPLOADED', 'PENDING', $3
		)`, id, sum, admin)
	if err != nil {
		t.Fatal(err)
	}
	return id.String()
}

func cleanupDoc(pool *pgxpool.Pool, id string) {
	ctx := context.Background()
	_, _ = pool.Exec(ctx, `UPDATE procedure_version_documents SET active_generation_id = NULL WHERE document_id = $1`, id)
	_, _ = pool.Exec(ctx, `DELETE FROM knowledge_chunks WHERE document_id = $1`, id)
	_, _ = pool.Exec(ctx, `DELETE FROM document_index_generations WHERE document_id = $1`, id)
	_, _ = pool.Exec(ctx, `DELETE FROM admin_write_idempotency WHERE request_id IN (SELECT request_id FROM document_index_jobs WHERE document_id = $1)`, id)
	_, _ = pool.Exec(ctx, `DELETE FROM audit_logs WHERE entity_id = $1`, id)
	_, _ = pool.Exec(ctx, `UPDATE documents SET supersedes_document_id = NULL WHERE supersedes_document_id = $1 OR id = $1`, id)
	_, _ = pool.Exec(ctx, `DELETE FROM document_index_jobs WHERE document_id = $1`, id)
	_, _ = pool.Exec(ctx, `DELETE FROM procedure_version_documents WHERE document_id = $1`, id)
	_, _ = pool.Exec(ctx, `DELETE FROM documents WHERE id = $1`, id)
}

func assertChunks(t *testing.T, pool *pgxpool.Pool, docID string, want int) {
	t.Helper()
	var n int
	if err := pool.QueryRow(context.Background(), `SELECT count(*) FROM knowledge_chunks WHERE document_id = $1`, docID).Scan(&n); err != nil {
		t.Fatal(err)
	}
	if n != want {
		t.Fatalf("chunks=%d", n)
	}
}
