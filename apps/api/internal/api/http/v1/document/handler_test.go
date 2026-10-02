package documentapi

import (
	"bytes"
	"context"
	"fmt"
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
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/document"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/middleware"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/storage"
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

func TestMain(m *testing.M) {
	gin.SetMode(gin.TestMode)
	os.Exit(m.Run())
}

type stubRepo struct {
	mu         sync.Mutex
	docs       []document.Document
	failInsert error
	inserts    int
}

func (s *stubRepo) Insert(_ context.Context, doc document.Document, _ uuid.UUID) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.inserts++
	if s.failInsert != nil {
		return s.failInsert
	}
	for _, existing := range s.docs {
		if existing.XAID == doc.XAID && existing.Checksum == doc.Checksum {
			return document.ErrDuplicate
		}
	}
	s.docs = append(s.docs, doc)
	return nil
}

func (s *stubRepo) Get(_ context.Context, xaID string, id uuid.UUID) (document.Document, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, doc := range s.docs {
		if doc.ID == id && doc.XAID == xaID {
			return doc, nil
		}
	}
	return document.Document{}, document.ErrNotFound
}

func (s *stubRepo) List(_ context.Context, f document.ListFilter) (document.ListResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var items []document.Document
	for _, doc := range s.docs {
		if doc.XAID != f.XAID {
			continue
		}
		if f.DomainID != "" && doc.DomainID != f.DomainID {
			continue
		}
		if f.ProcessingStatus != "" && doc.ProcessingStatus != f.ProcessingStatus {
			continue
		}
		if f.ValidityStatus != "" && doc.ValidityStatus != f.ValidityStatus {
			continue
		}
		items = append(items, doc)
	}
	if items == nil {
		items = []document.Document{}
	}
	return document.ListResult{Items: items, Count: len(items), Limit: f.Limit, Offset: f.Offset}, nil
}

func (s *stubRepo) DomainActive(_ context.Context, id string) error {
	if id == "ho_tich_chung_thuc" {
		return nil
	}
	return document.ErrDomain
}

func newEngine(t *testing.T, repo *stubRepo, objects *storage.Memory, max int64) (*gin.Engine, string, string) {
	t.Helper()
	dir := t.TempDir()
	svc := &document.Service{
		Repo: repo, Objects: objects, Bucket: "cas-documents", XAID: "xa_chu_se",
		MaxBytes: max, TempDir: dir, Logger: slog.New(slog.NewTextHandler(io.Discard, nil)),
	}
	tokens := &auth.TokenService{Secret: []byte("document-test-jwt-secret"), TTL: time.Hour}
	admin, _, err := tokens.Issue(uuid.MustParse("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"), auth.RoleAdmin, "a@example.com", "Admin")
	if err != nil {
		t.Fatal(err)
	}
	citizen, _, err := tokens.Issue(uuid.New(), auth.RoleCitizen, "c@example.com", "Citizen")
	if err != nil {
		t.Fatal(err)
	}
	r := gin.New()
	r.Use(middleware.RequestID())
	r.Use(middleware.ApiLog(slog.New(slog.NewTextHandler(io.Discard, nil))))
	h := NewHandler(svc)
	g := r.Group("/api/v1/admin/documents")
	g.Use(middleware.RequireAdmin(tokens))
	g.POST("", h.Upload)
	g.GET("", h.List)
	g.GET("/:id/content", h.Content)
	g.GET("/:id", h.Get)
	t.Cleanup(func() {
		entries, _ := os.ReadDir(dir)
		for _, e := range entries {
			if strings.HasPrefix(e.Name(), "cas-upload") {
				t.Errorf("temp upload left behind: %s", e.Name())
			}
		}
	})
	return r, admin, citizen
}

func pdfBytes(body string) []byte {
	return append([]byte("%PDF-1.4\n"), []byte(body)...)
}

func multipartUpload(t *testing.T, file []byte, filename, contentType string, fields map[string]string) (*bytes.Buffer, string) {
	t.Helper()
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)
	for k, v := range fields {
		if err := w.WriteField(k, v); err != nil {
			t.Fatal(err)
		}
	}
	hdr := make(textproto.MIMEHeader)
	hdr.Set("Content-Disposition", fmt.Sprintf(`form-data; name="file"; filename="%s"`, filename))
	if contentType != "" {
		hdr.Set("Content-Type", contentType)
	}
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
	return &buf, w.FormDataContentType()
}

func do(eng http.Handler, method, path, token, contentType string, body *bytes.Buffer, contentLength int64) *httptest.ResponseRecorder {
	var rdr io.Reader
	if body != nil {
		rdr = body
	}
	req := httptest.NewRequest(method, path, rdr)
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	if contentLength != 0 {
		req.ContentLength = contentLength
	}
	w := httptest.NewRecorder()
	eng.ServeHTTP(w, req)
	return w
}

func metaFields() map[string]string {
	return map[string]string{"title": "Hướng dẫn", "domain_id": "ho_tich_chung_thuc"}
}

func TestUploadRejectsMissingAndCitizenTokens(t *testing.T) {
	eng, _, citizen := newEngine(t, &stubRepo{}, storage.NewMemory(), 1024)
	body, ctype := multipartUpload(t, pdfBytes("a"), "a.pdf", "application/pdf", metaFields())
	if code := do(eng, http.MethodPost, "/api/v1/admin/documents", "", ctype, body, 0).Code; code != http.StatusUnauthorized {
		t.Fatalf("no token: %d", code)
	}
	body, ctype = multipartUpload(t, pdfBytes("a"), "a.pdf", "application/pdf", metaFields())
	if code := do(eng, http.MethodPost, "/api/v1/admin/documents", citizen, ctype, body, 0).Code; code != http.StatusForbidden {
		t.Fatalf("citizen: %d", code)
	}
}

func TestUploadAcceptsPDFAndListsIt(t *testing.T) {
	repo := &stubRepo{}
	objects := storage.NewMemory()
	eng, admin, _ := newEngine(t, repo, objects, 1024)
	body, ctype := multipartUpload(t, pdfBytes("hello"), "guide.pdf", "application/pdf", metaFields())
	res := do(eng, http.MethodPost, "/api/v1/admin/documents", admin, ctype, body, 0)
	if res.Code != http.StatusCreated {
		t.Fatalf("upload: %d %s", res.Code, res.Body.String())
	}
	if objects.Len() != 1 || repo.inserts != 1 {
		t.Fatalf("objects=%d inserts=%d", objects.Len(), repo.inserts)
	}
	if strings.Contains(res.Body.String(), "s3://") || strings.Contains(res.Body.String(), "cas-upload") {
		t.Fatalf("response leaked storage: %s", res.Body.String())
	}
	list := do(eng, http.MethodGet, "/api/v1/admin/documents?domain_id=ho_tich_chung_thuc&processing_status=UPLOADED&validity_status=PENDING", admin, "", nil, 0)
	if list.Code != http.StatusOK || !strings.Contains(list.Body.String(), "guide.pdf") {
		t.Fatalf("list: %d %s", list.Code, list.Body.String())
	}
}

func TestUploadRejectsBadFiles(t *testing.T) {
	eng, admin, _ := newEngine(t, &stubRepo{}, storage.NewMemory(), 1024)
	cases := []struct {
		name, filename, ctype string
		file                  []byte
	}{
		{"not pdf name", "a.txt", "application/pdf", pdfBytes("a")},
		{"traversal", "../a.pdf", "application/pdf", pdfBytes("a")},
		{"null", "a\x00.pdf", "application/pdf", pdfBytes("a")},
		{"empty name", "", "application/pdf", pdfBytes("a")},
		{"bad mime", "a.pdf", "text/plain", pdfBytes("a")},
		{"bad magic", "a.pdf", "application/pdf", []byte("hello world")},
		{"empty", "a.pdf", "application/pdf", nil},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			body, ctype := multipartUpload(t, tc.file, tc.filename, tc.ctype, metaFields())
			res := do(eng, http.MethodPost, "/api/v1/admin/documents", admin, ctype, body, 0)
			if res.Code != http.StatusUnprocessableEntity {
				t.Fatalf("status %d body %s", res.Code, res.Body.String())
			}
			if strings.Contains(res.Body.String(), "../") || strings.Contains(res.Body.String(), "\x00") {
				t.Fatalf("echoed filename: %s", res.Body.String())
			}
		})
	}
}

func TestUploadRejectsBadMetadata(t *testing.T) {
	eng, admin, _ := newEngine(t, &stubRepo{}, storage.NewMemory(), 1024)
	fields := metaFields()
	fields["title"] = ""
	body, ctype := multipartUpload(t, pdfBytes("a"), "a.pdf", "application/pdf", fields)
	if do(eng, http.MethodPost, "/api/v1/admin/documents", admin, ctype, body, 0).Code != http.StatusUnprocessableEntity {
		t.Fatal("empty title")
	}
	fields = metaFields()
	fields["domain_id"] = "nope"
	body, ctype = multipartUpload(t, pdfBytes("a"), "a.pdf", "application/pdf", fields)
	if do(eng, http.MethodPost, "/api/v1/admin/documents", admin, ctype, body, 0).Code != http.StatusUnprocessableEntity {
		t.Fatal("bad domain")
	}
	fields = metaFields()
	fields["effective_date"] = "32-13-99"
	body, ctype = multipartUpload(t, pdfBytes("a"), "a.pdf", "application/pdf", fields)
	if do(eng, http.MethodPost, "/api/v1/admin/documents", admin, ctype, body, 0).Code != http.StatusUnprocessableEntity {
		t.Fatal("bad date")
	}
}

func TestUploadTooLargeByLengthAndStream(t *testing.T) {
	eng, admin, _ := newEngine(t, &stubRepo{}, storage.NewMemory(), 32)
	body, ctype := multipartUpload(t, pdfBytes("a"), "a.pdf", "application/pdf", metaFields())
	res := do(eng, http.MethodPost, "/api/v1/admin/documents", admin, ctype, body, 32+middleware.DocumentUploadMultipartOverhead+10)
	if res.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("content-length: %d %s", res.Code, res.Body.String())
	}
	big := pdfBytes(strings.Repeat("x", 64))
	body, ctype = multipartUpload(t, big, "a.pdf", "application/pdf", metaFields())
	req := httptest.NewRequest(http.MethodPost, "/api/v1/admin/documents", body)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("Authorization", "Bearer "+admin)
	req.ContentLength = -1
	w := httptest.NewRecorder()
	eng.ServeHTTP(w, req)
	if w.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("chunked: %d %s", w.Code, w.Body.String())
	}
}

func TestDuplicateKeepsOneObject(t *testing.T) {
	repo := &stubRepo{}
	objects := storage.NewMemory()
	eng, admin, _ := newEngine(t, repo, objects, 1024)
	for i := 0; i < 2; i++ {
		body, ctype := multipartUpload(t, pdfBytes("same"), "a.pdf", "application/pdf", metaFields())
		_ = do(eng, http.MethodPost, "/api/v1/admin/documents", admin, ctype, body, 0)
	}
	body, ctype := multipartUpload(t, pdfBytes("same"), "a.pdf", "application/pdf", metaFields())
	res := do(eng, http.MethodPost, "/api/v1/admin/documents", admin, ctype, body, 0)
	if res.Code != http.StatusConflict || !strings.Contains(res.Body.String(), "DOCUMENT_DUPLICATE") {
		t.Fatalf("duplicate: %d %s", res.Code, res.Body.String())
	}
	if objects.Len() != 1 {
		t.Fatalf("objects=%d", objects.Len())
	}
}

func TestConcurrentUploadOneWinner(t *testing.T) {
	repo := &stubRepo{}
	objects := storage.NewMemory()
	eng, admin, _ := newEngine(t, repo, objects, 1024)
	var wg sync.WaitGroup
	codes := make(chan int, 2)
	type ready struct {
		body  *bytes.Buffer
		ctype string
	}
	reqs := make([]ready, 2)
	for i := range reqs {
		body, ctype := multipartUpload(t, pdfBytes("race"), "a.pdf", "application/pdf", metaFields())
		reqs[i] = ready{body, ctype}
	}
	for i := range reqs {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			codes <- do(eng, http.MethodPost, "/api/v1/admin/documents", admin, reqs[i].ctype, reqs[i].body, 0).Code
		}(i)
	}
	wg.Wait()
	close(codes)
	var created, conflict int
	for code := range codes {
		switch code {
		case http.StatusCreated:
			created++
		case http.StatusConflict:
			conflict++
		default:
			t.Fatalf("code %d", code)
		}
	}
	if created != 1 || conflict != 1 || objects.Len() != 1 {
		t.Fatalf("created=%d conflict=%d objects=%d", created, conflict, objects.Len())
	}
}

func TestStorageFailureDoesNotInsert(t *testing.T) {
	repo := &stubRepo{}
	objects := storage.NewMemory()
	objects.FailPut = true
	eng, admin, _ := newEngine(t, repo, objects, 1024)
	body, ctype := multipartUpload(t, pdfBytes("a"), "a.pdf", "application/pdf", metaFields())
	res := do(eng, http.MethodPost, "/api/v1/admin/documents", admin, ctype, body, 0)
	if res.Code != http.StatusInternalServerError || repo.inserts != 0 || objects.Len() != 0 {
		t.Fatalf("code=%d inserts=%d objects=%d", res.Code, repo.inserts, objects.Len())
	}
	if strings.Contains(res.Body.String(), "minio") || strings.Contains(res.Body.String(), "secret") {
		t.Fatalf("leaked: %s", res.Body.String())
	}
}

func TestInsertFailureDeletesObject(t *testing.T) {
	repo := &stubRepo{failInsert: io.ErrClosedPipe}
	objects := storage.NewMemory()
	eng, admin, _ := newEngine(t, repo, objects, 1024)
	body, ctype := multipartUpload(t, pdfBytes("a"), "a.pdf", "application/pdf", metaFields())
	res := do(eng, http.MethodPost, "/api/v1/admin/documents", admin, ctype, body, 0)
	if res.Code != http.StatusInternalServerError {
		t.Fatalf("code %d", res.Code)
	}
	if objects.Len() != 0 || objects.Deletes() == 0 {
		t.Fatalf("len=%d deletes=%d", objects.Len(), objects.Deletes())
	}
}

func TestDownloadAndMissingObject(t *testing.T) {
	repo := &stubRepo{}
	objects := storage.NewMemory()
	eng, admin, _ := newEngine(t, repo, objects, 1024)
	payload := pdfBytes("payload-body")
	body, ctype := multipartUpload(t, payload, "a.pdf", "application/pdf", metaFields())
	res := do(eng, http.MethodPost, "/api/v1/admin/documents", admin, ctype, body, 0)
	if res.Code != http.StatusCreated {
		t.Fatal(res.Body.String())
	}
	id := repo.docs[0].ID.String()
	got := do(eng, http.MethodGet, "/api/v1/admin/documents/"+id+"/content", admin, "", nil, 0)
	if got.Code != http.StatusOK || !bytes.Equal(got.Body.Bytes(), payload) {
		t.Fatalf("download %d %q", got.Code, got.Body.String())
	}
	other := do(eng, http.MethodGet, "/api/v1/admin/documents/"+uuid.NewString(), admin, "", nil, 0)
	if other.Code != http.StatusNotFound {
		t.Fatalf("missing id %d", other.Code)
	}
	key := storage.ObjectKey("xa_chu_se", id, repo.docs[0].Checksum)
	if err := objects.Delete(context.Background(), key); err != nil {
		t.Fatal(err)
	}
	missing := do(eng, http.MethodGet, "/api/v1/admin/documents/"+id+"/content", admin, "", nil, 0)
	if missing.Code != http.StatusNotFound {
		t.Fatalf("missing object %d %s", missing.Code, missing.Body.String())
	}
}

func TestImpossibleDatesReturn422(t *testing.T) {
	repo := &stubRepo{}
	eng, admin, _ := newEngine(t, repo, storage.NewMemory(), 1024)
	cases := []map[string]string{
		{"effective_date": "2026-02-31"},
		{"effective_date": "2026-13-01"},
		{"effective_date": "2026-02-29"},
		{"effective_date": "2026-03-01", "expire_date": "2026-02-28"},
	}
	for _, extra := range cases {
		fields := metaFields()
		for k, v := range extra {
			fields[k] = v
		}
		body, ctype := multipartUpload(t, pdfBytes("dates"), "a.pdf", "application/pdf", fields)
		res := do(eng, http.MethodPost, "/api/v1/admin/documents", admin, ctype, body, 0)
		if res.Code != http.StatusUnprocessableEntity {
			t.Fatalf("%v status %d body %s", extra, res.Code, res.Body.String())
		}
		if strings.Contains(res.Body.String(), "INTERNAL") {
			t.Fatalf("date error became 500-shaped: %s", res.Body.String())
		}
	}
	if repo.inserts != 0 {
		t.Fatalf("invalid dates reached insert: %d", repo.inserts)
	}
	fields := metaFields()
	fields["effective_date"] = "2024-02-29"
	fields["expire_date"] = "2024-03-01"
	body, ctype := multipartUpload(t, pdfBytes("ok-date"), "a.pdf", "application/pdf", fields)
	if res := do(eng, http.MethodPost, "/api/v1/admin/documents", admin, ctype, body, 0); res.Code != http.StatusCreated {
		t.Fatalf("real date: %d %s", res.Code, res.Body.String())
	}
}

func TestExactFileSizeAcceptedAndOneByteOver(t *testing.T) {
	exact := pdfBytes("exact-size")
	max := int64(len(exact))
	over := append(append([]byte{}, exact...), 'x')
	for _, chunked := range []bool{false, true} {
		t.Run(fmt.Sprintf("chunked=%v", chunked), func(t *testing.T) {
			repo := &stubRepo{}
			eng, admin, _ := newEngine(t, repo, storage.NewMemory(), max)
			ok := postUpload(t, eng, admin, exact, chunked)
			if ok.Code != http.StatusCreated {
				t.Fatalf("exact: %d %s", ok.Code, ok.Body.String())
			}
			big := postUpload(t, eng, admin, over, chunked)
			if big.Code != http.StatusRequestEntityTooLarge {
				t.Fatalf("plus one: %d %s", big.Code, big.Body.String())
			}
		})
	}
}

func TestHardMaxFileAcceptedWithMultipartOverhead(t *testing.T) {
	max := int64(middleware.DocumentUploadHardMax)
	for _, chunked := range []bool{false, true} {
		t.Run(fmt.Sprintf("chunked=%v", chunked), func(t *testing.T) {
			eng, admin, _ := newEngine(t, &stubRepo{}, storage.NewMemory(), max)
			okBody, okType := multipartFile(t, max, "ok.pdf")
			ok := postRaw(t, eng, admin, okBody, okType, chunked)
			if ok.Code != http.StatusCreated {
				t.Fatalf("exact hard max: %d %s", ok.Code, ok.Body.String())
			}
			bigBody, bigType := multipartFile(t, max+1, "big.pdf")
			big := postRaw(t, eng, admin, bigBody, bigType, chunked)
			if big.Code != http.StatusRequestEntityTooLarge {
				t.Fatalf("hard max+1: %d %s", big.Code, big.Body.String())
			}
		})
	}
}

func postUpload(t *testing.T, eng http.Handler, token string, file []byte, chunked bool) *httptest.ResponseRecorder {
	t.Helper()
	body, ctype := multipartUpload(t, file, "a.pdf", "application/pdf", metaFields())
	length := int64(body.Len())
	if chunked {
		length = -1
	}
	return do(eng, http.MethodPost, "/api/v1/admin/documents", token, ctype, body, length)
}

type pdfStream struct {
	n   int64
	off int64
}

func (p *pdfStream) Read(b []byte) (int, error) {
	if p.off >= p.n {
		return 0, io.EOF
	}
	if int64(len(b)) > p.n-p.off {
		b = b[:p.n-p.off]
	}
	for i := range b {
		b[i] = 'x'
	}
	if p.off == 0 {
		copy(b, []byte("%PDF-1.4\n"))
	}
	p.off += int64(len(b))
	return len(b), nil
}

func multipartFile(t *testing.T, size int64, filename string) (*os.File, string) {
	t.Helper()
	f, err := os.CreateTemp(t.TempDir(), "upload-body-*")
	if err != nil {
		t.Fatal(err)
	}
	w := multipart.NewWriter(f)
	for k, v := range metaFields() {
		if err := w.WriteField(k, v); err != nil {
			t.Fatal(err)
		}
	}
	hdr := make(textproto.MIMEHeader)
	hdr.Set("Content-Disposition", fmt.Sprintf(`form-data; name="file"; filename="%s"`, filename))
	hdr.Set("Content-Type", "application/pdf")
	part, err := w.CreatePart(hdr)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := io.Copy(part, &pdfStream{n: size}); err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		t.Fatal(err)
	}
	return f, w.FormDataContentType()
}

func postRaw(t *testing.T, eng http.Handler, token string, body *os.File, contentType string, chunked bool) *httptest.ResponseRecorder {
	t.Helper()
	info, err := body.Stat()
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, "/api/v1/admin/documents", body)
	req.Header.Set("Content-Type", contentType)
	req.Header.Set("Authorization", "Bearer "+token)
	if chunked {
		req.ContentLength = -1
	} else {
		req.ContentLength = info.Size()
	}
	w := httptest.NewRecorder()
	eng.ServeHTTP(w, req)
	return w
}
