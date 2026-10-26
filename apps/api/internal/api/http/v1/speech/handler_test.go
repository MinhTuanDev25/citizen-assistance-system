package speechapi

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/middleware"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/repository"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/speech"
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

const translatePath = "/api/v1/sessions/:sessionId/speech/translate"

func TestTranslateSuccessAndAudit(t *testing.T) {
	gin.SetMode(gin.TestMode)
	sess := guestSession()
	client := &fakeTranslator{result: okResult()}
	audit := &fakeAudit{}
	h := testHandler(true, client, sess, audit)
	res := postTranslate(t, h, sess, wavBytes(), map[string]string{
		HeaderGuestToken: *sess.GuestToken, middleware.HeaderRequestID: uuid.NewString(),
	})
	if res.Code != http.StatusOK {
		t.Fatalf("status %d body %s", res.Code, res.Body.String())
	}
	if client.calls.Load() != 1 {
		t.Fatalf("calls %d", client.calls.Load())
	}
	if !strings.Contains(res.Body.String(), `"translated_text_vi":"tôi"`) {
		t.Fatal(res.Body.String())
	}
	if strings.Contains(res.Body.String(), "tokenizer_parity") || strings.Contains(res.Body.String(), "/secret") {
		t.Fatal("leaked metadata")
	}
	if len(audit.recs) != 1 || audit.recs[0].Outcome != speech.OutcomeSuccess {
		t.Fatalf("audit %+v", audit.recs)
	}
	if audit.recs[0].ResearchProvenance != "NOT_VERIFIED" {
		t.Fatal("must copy provenance, not invent VERIFIED")
	}
	if audit.recs[0].XaID != sess.XaID {
		t.Fatalf("audit xa_id %q", audit.recs[0].XaID)
	}
}

func TestUnauthorizedDoesNotBufferAudio(t *testing.T) {
	gin.SetMode(gin.TestMode)
	sess := guestSession()
	client := &fakeTranslator{result: okResult()}
	h := testHandler(true, client, sess, &fakeAudit{})
	audio := append(wavBytes(), bytes.Repeat([]byte("a"), 32<<10)...)
	n, res := postTranslateCount(t, h, sess, audio, nil)
	assertCode(t, res, http.StatusUnauthorized, "UNAUTHORIZED")
	if n.Load() != 0 {
		t.Fatalf("buffered %d bytes before 401", n.Load())
	}
	if client.calls.Load() != 0 {
		t.Fatal("upstream called")
	}
}

func TestInvalidGuestTokenDoesNotBufferAudio(t *testing.T) {
	gin.SetMode(gin.TestMode)
	sess := guestSession()
	h := testHandler(true, &fakeTranslator{result: okResult()}, sess, &fakeAudit{})
	n, res := postTranslateCount(t, h, sess, wavBytes(), map[string]string{HeaderGuestToken: "nope"})
	assertCode(t, res, http.StatusForbidden, "FORBIDDEN")
	if n.Load() != 0 {
		t.Fatalf("buffered %d bytes before 403", n.Load())
	}
}

func TestInvalidSessionID(t *testing.T) {
	gin.SetMode(gin.TestMode)
	sess := guestSession()
	h := testHandler(true, &fakeTranslator{result: okResult()}, sess, &fakeAudit{})
	engine := gin.New()
	engine.Use(middleware.RequestID())
	engine.POST(translatePath, h.Translate)
	req := mustMultipart(wavBytes(), "not-a-uuid", map[string]string{HeaderGuestToken: *sess.GuestToken})
	res := httptest.NewRecorder()
	engine.ServeHTTP(res, req)
	assertCode(t, res, http.StatusBadRequest, "VALIDATION")
}

func TestClosedSessionDoesNotBufferAudio(t *testing.T) {
	gin.SetMode(gin.TestMode)
	sess := guestSession()
	sess.Status = "COMPLETED"
	h := testHandler(true, &fakeTranslator{result: okResult()}, sess, &fakeAudit{})
	n, res := postTranslateCount(t, h, sess, wavBytes(), map[string]string{HeaderGuestToken: *sess.GuestToken})
	assertCode(t, res, http.StatusConflict, "CONFLICT")
	if n.Load() != 0 {
		t.Fatalf("buffered %d bytes for closed session", n.Load())
	}
}

func TestJWTCannotAccessAnotherSession(t *testing.T) {
	gin.SetMode(gin.TestMode)
	owner := uuid.New()
	other := uuid.New()
	sess := &repository.Session{ID: uuid.New(), UserID: &owner, XaID: "xa_chu_se", Status: "OPEN"}
	h := testHandler(true, &fakeTranslator{result: okResult()}, sess, &fakeAudit{})
	engine := gin.New()
	engine.Use(middleware.RequestID())
	engine.Use(func(c *gin.Context) { c.Set(middleware.ContextUserID, other.String()); c.Next() })
	engine.POST(translatePath, h.Translate)
	n := &atomic.Int64{}
	req := mustMultipartCounted(wavBytes(), sess.ID.String(), nil, n)
	res := httptest.NewRecorder()
	engine.ServeHTTP(res, req)
	assertCode(t, res, http.StatusForbidden, "FORBIDDEN")
	if n.Load() != 0 {
		t.Fatalf("buffered %d bytes", n.Load())
	}
}

func TestUnknownSessionIsNotCrossCommuneEvidence(t *testing.T) {
	gin.SetMode(gin.TestMode)
	sess := guestSession()
	h := testHandler(true, &fakeTranslator{result: okResult()}, sess, &fakeAudit{})
	other := uuid.New()
	engine := gin.New()
	engine.Use(middleware.RequestID())
	engine.POST(translatePath, h.Translate)
	req := mustMultipart(wavBytes(), other.String(), map[string]string{HeaderGuestToken: *sess.GuestToken})
	res := httptest.NewRecorder()
	engine.ServeHTTP(res, req)
	assertCode(t, res, http.StatusNotFound, "NOT_FOUND")
}

func TestCrossCommuneGuestCannotUseOtherSession(t *testing.T) {
	gin.SetMode(gin.TestMode)
	a := guestSession()
	a.XaID = "xa_a"
	bTok := "guest-token-b"
	b := &repository.Session{ID: uuid.New(), GuestToken: &bTok, XaID: "xa_b", Status: "OPEN"}
	client := &fakeTranslator{result: okResult()}
	audit := &fakeAudit{}
	h := NewHandler(Handler{
		Enabled: true, Client: client, Sessions: sessionsOf(a, b), Audit: audit,
		MaxUploadBytes: 1024, MaxInflight: 1,
		Logger: slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	n, res := postTranslateCount(t, h, b, wavBytes(), map[string]string{HeaderGuestToken: *a.GuestToken})
	assertCode(t, res, http.StatusForbidden, "FORBIDDEN")
	if n.Load() != 0 {
		t.Fatalf("buffered %d bytes", n.Load())
	}
	if client.calls.Load() != 0 {
		t.Fatal("upstream called")
	}
	if len(audit.recs) != 0 {
		t.Fatalf("audit leaked to wrong commune: %+v", audit.recs)
	}
}

func TestCrossCommuneJWTCannotUseOtherUserSession(t *testing.T) {
	gin.SetMode(gin.TestMode)
	userA, userB := uuid.New(), uuid.New()
	sessB := &repository.Session{ID: uuid.New(), UserID: &userB, XaID: "xa_b", Status: "OPEN"}
	client := &fakeTranslator{result: okResult()}
	h := NewHandler(Handler{
		Enabled: true, Client: client, Sessions: sessionsOf(sessB),
		Audit: &fakeAudit{}, MaxUploadBytes: 1024, MaxInflight: 1,
		Logger: slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	engine := gin.New()
	engine.Use(middleware.RequestID())
	engine.Use(func(c *gin.Context) { c.Set(middleware.ContextUserID, userA.String()); c.Next() })
	engine.POST(translatePath, h.Translate)
	n := &atomic.Int64{}
	req := mustMultipartCounted(wavBytes(), sessB.ID.String(), nil, n)
	res := httptest.NewRecorder()
	engine.ServeHTTP(res, req)
	assertCode(t, res, http.StatusForbidden, "FORBIDDEN")
	if n.Load() != 0 || client.calls.Load() != 0 {
		t.Fatalf("buffer=%d calls=%d", n.Load(), client.calls.Load())
	}
}

func TestChatSpeechAuthParitySource(t *testing.T) {
	chatSrc, err := os.ReadFile(filepath.Join("..", "session", "handler.go"))
	if err != nil {
		t.Fatal(err)
	}
	speechSrc, err := os.ReadFile("handler.go")
	if err != nil {
		t.Fatal(err)
	}
	needles := []string{
		`response.Fail(c, http.StatusForbidden, "FORBIDDEN", "session belongs to another user")`,
		`response.Fail(c, http.StatusUnauthorized, "UNAUTHORIZED", "X-Guest-Token header is required")`,
		`response.Fail(c, http.StatusForbidden, "FORBIDDEN", "guest token mismatch")`,
		`if sess.UserID != nil {`,
		`uid, ok := middleware.UserID(c)`,
	}
	for _, n := range needles {
		if !bytes.Contains(chatSrc, []byte(n)) || !bytes.Contains(speechSrc, []byte(n)) {
			t.Fatalf("auth branch missing from Chat or Speech: %s", n)
		}
	}
	if bytes.Contains(chatSrc, []byte("sess.XaID != cfg.XAID")) || bytes.Contains(speechSrc, []byte("sess.XaID != cfg.XAID")) {
		t.Fatal("turn/speech auth must not add a deployment commune pin gate")
	}
}

func TestSameUserMayOwnSessionsInTwoCommunes(t *testing.T) {
	gin.SetMode(gin.TestMode)
	uid := uuid.New()
	sessA := &repository.Session{ID: uuid.New(), UserID: &uid, XaID: "xa_a", Status: "OPEN"}
	sessB := &repository.Session{ID: uuid.New(), UserID: &uid, XaID: "xa_b", Status: "OPEN"}
	audit := &fakeAudit{}
	h := NewHandler(Handler{
		Enabled: true, Client: &fakeTranslator{result: okResult()},
		Sessions: sessionsOf(sessA, sessB), Audit: audit,
		MaxUploadBytes: 1024, MaxInflight: 1,
		Logger: slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	engine := gin.New()
	engine.Use(middleware.RequestID())
	engine.Use(func(c *gin.Context) { c.Set(middleware.ContextUserID, uid.String()); c.Next() })
	engine.POST(translatePath, h.Translate)
	res := httptest.NewRecorder()
	engine.ServeHTTP(res, mustMultipart(wavBytes(), sessB.ID.String(), nil))
	if res.Code != http.StatusOK {
		t.Fatalf("%d %s", res.Code, res.Body.String())
	}
	if len(audit.recs) != 1 || audit.recs[0].XaID != "xa_b" {
		t.Fatalf("audit xa %+v", audit.recs)
	}
}

func TestTranslateDisabledDoesNotCallUpstreamOrBuffer(t *testing.T) {
	gin.SetMode(gin.TestMode)
	sess := guestSession()
	client := &fakeTranslator{result: okResult()}
	h := testHandler(false, client, sess, &fakeAudit{})
	n, res := postTranslateCount(t, h, sess, wavBytes(), map[string]string{HeaderGuestToken: *sess.GuestToken})
	assertCode(t, res, http.StatusServiceUnavailable, "SPEECH_DISABLED")
	if client.calls.Load() != 0 {
		t.Fatal("disabled still called upstream")
	}
	if n.Load() != 0 {
		t.Fatalf("buffered %d bytes while disabled", n.Load())
	}
}

func TestAdmissionRejectsBeforeBuffering(t *testing.T) {
	gin.SetMode(gin.TestMode)
	sess := guestSession()
	started := make(chan struct{})
	release := make(chan struct{})
	h := testHandler(true, &fakeTranslator{result: okResult()}, sess, &fakeAudit{})
	h.Client = &blockingTranslator{started: started, release: release, result: okResult()}
	engine := testEngine(h, sess)

	first := mustMultipart(wavBytes(), sess.ID.String(), map[string]string{HeaderGuestToken: *sess.GuestToken})
	done := make(chan *httptest.ResponseRecorder, 1)
	go func() {
		res := httptest.NewRecorder()
		engine.ServeHTTP(res, first)
		done <- res
	}()
	select {
	case <-started:
	case <-time.After(2 * time.Second):
		t.Fatal("first request did not start inference")
	}
	n := &atomic.Int64{}
	secondReq := mustMultipartCounted(append(wavBytes(), bytes.Repeat([]byte("b"), 4096)...), sess.ID.String(), map[string]string{HeaderGuestToken: *sess.GuestToken}, n)
	second := httptest.NewRecorder()
	engine.ServeHTTP(second, secondReq)
	assertCode(t, second, http.StatusServiceUnavailable, "SPEECH_OVERLOADED")
	if n.Load() != 0 {
		t.Fatalf("second request buffered %d bytes while overloaded", n.Load())
	}
	close(release)
	firstRes := <-done
	if firstRes.Code != http.StatusOK {
		t.Fatalf("first %d %s", firstRes.Code, firstRes.Body.String())
	}
}

func TestSlotReleaseOnValidationAndUpstreamFailure(t *testing.T) {
	gin.SetMode(gin.TestMode)
	sess := guestSession()
	h := testHandler(true, &fakeTranslator{err: speech.ErrUnavailable}, sess, &fakeAudit{})
	res := postTranslate(t, h, sess, []byte("ID3junk"), map[string]string{HeaderGuestToken: *sess.GuestToken})
	assertCode(t, res, http.StatusBadRequest, "VALIDATION")
	h.Client = &fakeTranslator{result: okResult()}
	ok := postTranslate(t, h, sess, wavBytes(), map[string]string{HeaderGuestToken: *sess.GuestToken})
	if ok.Code != http.StatusOK {
		t.Fatalf("slot not released after validation failure: %d %s", ok.Code, ok.Body.String())
	}
}

func TestSlotReleaseOnUpstreamFailureThenSuccess(t *testing.T) {
	gin.SetMode(gin.TestMode)
	sess := guestSession()
	h := testHandler(true, &fakeTranslator{err: speech.ErrUnavailable}, sess, &fakeAudit{})
	res := postTranslate(t, h, sess, wavBytes(), map[string]string{HeaderGuestToken: *sess.GuestToken})
	assertCode(t, res, http.StatusBadGateway, "SPEECH_UNAVAILABLE")
	h.Client = &fakeTranslator{result: okResult()}
	ok := postTranslate(t, h, sess, wavBytes(), map[string]string{HeaderGuestToken: *sess.GuestToken})
	if ok.Code != http.StatusOK {
		t.Fatalf("slot not released after upstream failure: %d", ok.Code)
	}
}

func TestTranslateMapsUpstreamErrors(t *testing.T) {
	gin.SetMode(gin.TestMode)
	sess := guestSession()
	table := []struct {
		err  error
		code string
		http int
	}{
		{speech.ErrUnavailable, "SPEECH_UNAVAILABLE", http.StatusBadGateway},
		{speech.ErrNoSpeech, "SPEECH_NO_SPEECH", http.StatusUnprocessableEntity},
		{speech.ErrOverloaded, "SPEECH_OVERLOADED", http.StatusServiceUnavailable},
		{speech.ErrTimeout, "SPEECH_TIMEOUT", http.StatusGatewayTimeout},
		{speech.ErrInvalidResponse, "SPEECH_INVALID_UPSTREAM", http.StatusBadGateway},
		{speech.ErrUpstreamAuth, "SPEECH_UPSTREAM_AUTH", http.StatusBadGateway},
		{speech.ErrCanceled, "SPEECH_CANCELLED", http.StatusBadRequest},
	}
	for _, tc := range table {
		h := testHandler(true, &fakeTranslator{err: tc.err}, sess, &fakeAudit{})
		res := postTranslate(t, h, sess, wavBytes(), map[string]string{HeaderGuestToken: *sess.GuestToken})
		assertCode(t, res, tc.http, tc.code)
		if strings.Contains(res.Body.String(), "Traceback") || strings.Contains(res.Body.String(), "/workspace") {
			t.Fatal(res.Body.String())
		}
	}
}

func TestNewHandlerDoesNotFailOpenUploadLimit(t *testing.T) {
	gin.SetMode(gin.TestMode)
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	zero := NewHandler(Handler{MaxUploadBytes: 0, Logger: log})
	if zero.MaxUploadBytes != 0 {
		t.Fatalf("zero became %d", zero.MaxUploadBytes)
	}
	neg := NewHandler(Handler{MaxUploadBytes: -1, Logger: log})
	if neg.MaxUploadBytes != -1 {
		t.Fatalf("negative became %d", neg.MaxUploadBytes)
	}
	kept := NewHandler(Handler{MaxUploadBytes: 1024, Logger: log})
	if kept.MaxUploadBytes != 1024 {
		t.Fatalf("explicit cap rewritten to %d", kept.MaxUploadBytes)
	}
}

func TestZeroUploadLimitRejectsWithoutUpstream(t *testing.T) {
	gin.SetMode(gin.TestMode)
	sess := guestSession()
	client := &fakeTranslator{result: okResult()}
	h := testHandler(true, client, sess, &fakeAudit{})
	h.MaxUploadBytes = 0
	n, res := postTranslateCount(t, h, sess, wavBytes(), map[string]string{HeaderGuestToken: *sess.GuestToken})
	assertCode(t, res, http.StatusRequestEntityTooLarge, "PAYLOAD_TOO_LARGE")
	if client.calls.Load() != 0 {
		t.Fatal("upstream called")
	}
	if n.Load() != 0 {
		t.Fatalf("buffered %d bytes after zero cap", n.Load())
	}
}

func TestConfiguredBodyLimitRejectsEncodedMultipart(t *testing.T) {
	gin.SetMode(gin.TestMode)
	sess := guestSession()
	client := &fakeTranslator{result: okResult()}
	h := testHandler(true, client, sess, &fakeAudit{})
	h.MaxUploadBytes = 1024
	h.MaxBodyBytes = 64
	n, res := postTranslateCount(t, h, sess, wavBytes(), map[string]string{HeaderGuestToken: *sess.GuestToken})
	assertCode(t, res, http.StatusRequestEntityTooLarge, "PAYLOAD_TOO_LARGE")
	if client.calls.Load() != 0 {
		t.Fatal("upstream called")
	}
	_ = n
}

func TestTranslateOversizedAndChunked(t *testing.T) {
	gin.SetMode(gin.TestMode)
	sess := guestSession()
	h := testHandler(true, &fakeTranslator{result: okResult()}, sess, &fakeAudit{})
	h.MaxUploadBytes = 16
	res := postTranslate(t, h, sess, append(wavBytes(), bytes.Repeat([]byte("a"), 64)...), map[string]string{HeaderGuestToken: *sess.GuestToken})
	assertCode(t, res, http.StatusRequestEntityTooLarge, "PAYLOAD_TOO_LARGE")

	engine := testEngine(h, sess)
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)
	part, err := w.CreateFormFile("audio", "utt.wav")
	if err != nil {
		t.Fatal(err)
	}
	_, _ = part.Write(append(wavBytes(), bytes.Repeat([]byte("c"), 64)...))
	_ = w.Close()
	req := httptest.NewRequest(http.MethodPost, "/api/v1/sessions/"+sess.ID.String()+"/speech/translate", &buf)
	req.ContentLength = -1
	req.Header.Set("Content-Type", w.FormDataContentType())
	req.Header.Set(HeaderGuestToken, *sess.GuestToken)
	chunked := httptest.NewRecorder()
	engine.ServeHTTP(chunked, req)
	assertCode(t, chunked, http.StatusRequestEntityTooLarge, "PAYLOAD_TOO_LARGE")
}

func TestTranslateMissingEmptyAudio(t *testing.T) {
	gin.SetMode(gin.TestMode)
	sess := guestSession()
	h := testHandler(true, &fakeTranslator{result: okResult()}, sess, &fakeAudit{})
	res := postTranslate(t, h, sess, nil, map[string]string{HeaderGuestToken: *sess.GuestToken})
	assertCode(t, res, http.StatusBadRequest, "VALIDATION")
	res = postTranslate(t, h, sess, []byte{}, map[string]string{HeaderGuestToken: *sess.GuestToken})
	assertCode(t, res, http.StatusBadRequest, "VALIDATION")
}

func TestTranslatePropagatesRequestID(t *testing.T) {
	gin.SetMode(gin.TestMode)
	sess := guestSession()
	rid := "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
	client := &fakeTranslator{result: okResult()}
	h := testHandler(true, client, sess, &fakeAudit{})
	res := postTranslate(t, h, sess, wavBytes(), map[string]string{
		HeaderGuestToken: *sess.GuestToken, middleware.HeaderRequestID: rid,
	})
	if res.Code != 200 {
		t.Fatal(res.Body.String())
	}
	if client.last.RequestID != rid {
		t.Fatalf("upstream rid %s", client.last.RequestID)
	}
}

func TestTranslateLoggedInSession(t *testing.T) {
	gin.SetMode(gin.TestMode)
	uid := uuid.New()
	sess := &repository.Session{ID: uuid.New(), UserID: &uid, XaID: "xa_chu_se", Status: "OPEN"}
	h := testHandler(true, &fakeTranslator{result: okResult()}, sess, &fakeAudit{})
	engine := gin.New()
	engine.Use(middleware.RequestID())
	engine.Use(func(c *gin.Context) { c.Set(middleware.ContextUserID, uid.String()); c.Next() })
	engine.POST(translatePath, h.Translate)
	res := httptest.NewRecorder()
	engine.ServeHTTP(res, mustMultipart(wavBytes(), sess.ID.String(), nil))
	if res.Code != 200 {
		t.Fatalf("%d %s", res.Code, res.Body.String())
	}
}

func TestBestEffortAuditFailureStill200(t *testing.T) {
	gin.SetMode(gin.TestMode)
	sess := guestSession()
	logs := &logSink{}
	h := testHandler(true, &fakeTranslator{result: okResult()}, sess, &failAudit{err: errors.New("db down")})
	h.Logger = slog.New(slog.NewTextHandler(logs, nil))
	res := postTranslate(t, h, sess, wavBytes(), map[string]string{HeaderGuestToken: *sess.GuestToken})
	if res.Code != http.StatusOK {
		t.Fatalf("best-effort must keep 200, got %d %s", res.Code, res.Body.String())
	}
	if !strings.Contains(logs.String(), "speech_audit_failed") {
		t.Fatalf("audit failure not logged: %s", logs.String())
	}
	if strings.Contains(logs.String(), *sess.GuestToken) {
		t.Fatal("token in audit log")
	}
}

func TestNilAuditIsObservable(t *testing.T) {
	gin.SetMode(gin.TestMode)
	sess := guestSession()
	logs := &logSink{}
	h := testHandler(true, &fakeTranslator{result: okResult()}, sess, nil)
	h.Audit = nil
	h.Logger = slog.New(slog.NewTextHandler(logs, nil))
	res := postTranslate(t, h, sess, wavBytes(), map[string]string{HeaderGuestToken: *sess.GuestToken})
	if res.Code != http.StatusOK {
		t.Fatalf("got %d", res.Code)
	}
	if !strings.Contains(logs.String(), "speech_audit_failed") {
		t.Fatalf("missing audit log: %s", logs.String())
	}
}

func TestTranslateNoSpeechIsNotUnavailable(t *testing.T) {
	gin.SetMode(gin.TestMode)
	sess := guestSession()
	audit := &fakeAudit{}
	h := testHandler(true, &fakeTranslator{err: speech.ErrNoSpeech}, sess, audit)
	res := postTranslate(t, h, sess, wavBytes(), map[string]string{HeaderGuestToken: *sess.GuestToken})
	assertCode(t, res, http.StatusUnprocessableEntity, "SPEECH_NO_SPEECH")
	if strings.Contains(res.Body.String(), "SPEECH_UNAVAILABLE") {
		t.Fatal(res.Body.String())
	}
	if len(audit.recs) != 1 {
		t.Fatalf("audit %+v", audit.recs)
	}
	rec := audit.recs[0]
	if rec.Outcome != speech.OutcomeFailure || rec.ErrorCode != "SPEECH_NO_SPEECH" || rec.HTTPStatus != http.StatusUnprocessableEntity {
		t.Fatalf("audit %+v", rec)
	}
}

func TestCancelledRequestStillAttemptsAudit(t *testing.T) {
	gin.SetMode(gin.TestMode)
	sess := guestSession()
	audit := &captureAudit{}
	h := testHandler(true, &fakeTranslator{err: speech.ErrCanceled}, sess, audit)
	res := postTranslate(t, h, sess, wavBytes(), map[string]string{HeaderGuestToken: *sess.GuestToken})
	assertCode(t, res, http.StatusBadRequest, "SPEECH_CANCELLED")
	if len(audit.recs) != 1 || audit.recs[0].Outcome != speech.OutcomeFailure {
		t.Fatalf("audit %+v", audit.recs)
	}
	if audit.sawCancel {
		t.Fatal("audit context inherited HTTP cancellation")
	}
	if !audit.sawDeadline {
		t.Fatal("audit context missing deadline")
	}
}

func TestAuditTimeout(t *testing.T) {
	gin.SetMode(gin.TestMode)
	sess := guestSession()
	audit := &delayAudit{delay: 200 * time.Millisecond}
	h := testHandler(true, &fakeTranslator{result: okResult()}, sess, audit)
	h.AuditTimeout = 20 * time.Millisecond
	res := postTranslate(t, h, sess, wavBytes(), map[string]string{HeaderGuestToken: *sess.GuestToken})
	if res.Code != http.StatusOK {
		t.Fatalf("got %d", res.Code)
	}
	if !errors.Is(audit.ctxErr, context.DeadlineExceeded) {
		t.Fatalf("want deadline, got %v", audit.ctxErr)
	}
}

func TestLegacyRouteGoneWithoutBuffering(t *testing.T) {
	gin.SetMode(gin.TestMode)
	h := NewHandler(Handler{Logger: slog.New(slog.NewTextHandler(io.Discard, nil))})
	engine := gin.New()
	engine.POST("/api/v1/speech/translate", h.GoneLegacy)
	n := &atomic.Int64{}
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)
	part, _ := w.CreateFormFile("audio", "utt.wav")
	_, _ = part.Write(bytes.Repeat([]byte("z"), 4096))
	_ = w.Close()
	req := httptest.NewRequest(http.MethodPost, "/api/v1/speech/translate", &countingReader{Reader: bytes.NewReader(buf.Bytes()), n: n})
	req.Header.Set("Content-Type", w.FormDataContentType())
	res := httptest.NewRecorder()
	engine.ServeHTTP(res, req)
	assertCode(t, res, http.StatusGone, "DEPRECATED")
	if n.Load() != 0 {
		t.Fatalf("legacy route buffered %d bytes", n.Load())
	}
}

func testHandler(enabled bool, client speech.Translator, sess *repository.Session, audit Auditor) *Handler {
	return NewHandler(Handler{
		Enabled:        enabled,
		Client:         client,
		Sessions:       sessionsOf(sess),
		Audit:          audit,
		MaxUploadBytes: 1024,
		MaxInflight:    1,
		Logger:         slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
}

func testEngine(h *Handler, _ *repository.Session) *gin.Engine {
	engine := gin.New()
	engine.Use(middleware.RequestID())
	engine.POST(translatePath, h.Translate)
	return engine
}

func postTranslate(t *testing.T, h *Handler, sess *repository.Session, audio []byte, headers map[string]string) *httptest.ResponseRecorder {
	t.Helper()
	engine := testEngine(h, sess)
	res := httptest.NewRecorder()
	engine.ServeHTTP(res, mustMultipart(audio, sess.ID.String(), headers))
	return res
}

func postTranslateCount(t *testing.T, h *Handler, sess *repository.Session, audio []byte, headers map[string]string) (*atomic.Int64, *httptest.ResponseRecorder) {
	t.Helper()
	n := &atomic.Int64{}
	engine := testEngine(h, sess)
	res := httptest.NewRecorder()
	engine.ServeHTTP(res, mustMultipartCounted(audio, sess.ID.String(), headers, n))
	return n, res
}

func mustMultipart(audio []byte, sessionID string, headers map[string]string) *http.Request {
	return mustMultipartCounted(audio, sessionID, headers, nil)
}

func mustMultipartCounted(audio []byte, sessionID string, headers map[string]string, n *atomic.Int64) *http.Request {
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)
	if audio != nil {
		part, err := w.CreateFormFile("audio", "utt.wav")
		if err != nil {
			panic(err)
		}
		_, _ = part.Write(audio)
	}
	_ = w.Close()
	url := "/api/v1/sessions/" + sessionID + "/speech/translate"
	var body io.Reader = bytes.NewReader(buf.Bytes())
	if n != nil {
		body = &countingReader{Reader: bytes.NewReader(buf.Bytes()), n: n}
	}
	req := httptest.NewRequest(http.MethodPost, url, body)
	req.Header.Set("Content-Type", w.FormDataContentType())
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	return req
}

func assertCode(t *testing.T, res *httptest.ResponseRecorder, status int, code string) {
	t.Helper()
	if res.Code != status {
		t.Fatalf("status %d want %d body %s", res.Code, status, res.Body.String())
	}
	var env struct {
		Error *struct{ Code string } `json:"error"`
	}
	_ = json.Unmarshal(res.Body.Bytes(), &env)
	if env.Error == nil || env.Error.Code != code {
		t.Fatalf("code %v want %s body %s", env.Error, code, res.Body.String())
	}
}

func guestSession() *repository.Session {
	tok := "guest-token-value"
	return &repository.Session{ID: uuid.New(), GuestToken: &tok, XaID: "xa_chu_se", Status: "OPEN"}
}

func wavBytes() []byte {
	b := make([]byte, 16)
	copy(b[0:4], []byte("RIFF"))
	copy(b[8:12], []byte("WAVE"))
	return b
}

func okResult() *speech.Result {
	flag := true
	return &speech.Result{
		ModelID: "c0_cascaded", AsrTextBahnar: "xin", TranslatedTextVi: "tôi",
		DurationMS: 10, LatencyMS: 7, TokenizerParity: "VERIFIED",
		ResearchProvenance: "NOT_VERIFIED", CheckpointIdentity: "expected_step_match",
		GenerationMatchesRQ1: &flag, UpstreamStatus: 200,
	}
}

type countingReader struct {
	io.Reader
	n *atomic.Int64
}

func (c *countingReader) Read(p []byte) (int, error) {
	n, err := c.Reader.Read(p)
	c.n.Add(int64(n))
	return n, err
}

type blockingTranslator struct {
	started chan struct{}
	release chan struct{}
	result  *speech.Result
}

func (b *blockingTranslator) Infer(_ context.Context, _ speech.InferRequest) (*speech.Result, error) {
	close(b.started)
	<-b.release
	return b.result, nil
}

type fakeTranslator struct {
	calls  atomic.Int32
	last   speech.InferRequest
	result *speech.Result
	err    error
}

func (f *fakeTranslator) Infer(_ context.Context, req speech.InferRequest) (*speech.Result, error) {
	f.calls.Add(1)
	f.last = req
	if f.err != nil {
		return nil, f.err
	}
	return f.result, nil
}

type fakeSessions struct {
	byID map[uuid.UUID]*repository.Session
}

func sessionsOf(ss ...*repository.Session) *fakeSessions {
	m := make(map[uuid.UUID]*repository.Session, len(ss))
	for _, s := range ss {
		m[s.ID] = s
	}
	return &fakeSessions{byID: m}
}

func (f *fakeSessions) GetByID(_ context.Context, id uuid.UUID) (*repository.Session, error) {
	if f == nil || f.byID == nil {
		return nil, nil
	}
	return f.byID[id], nil
}

type fakeAudit struct {
	mu   sync.Mutex
	recs []speech.Record
}

func (f *fakeAudit) Insert(_ context.Context, rec speech.Record) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.recs = append(f.recs, rec)
	return nil
}

type failAudit struct{ err error }

func (f *failAudit) Insert(_ context.Context, _ speech.Record) error { return f.err }

type captureAudit struct {
	recs        []speech.Record
	sawCancel   bool
	sawDeadline bool
}

func (c *captureAudit) Insert(ctx context.Context, rec speech.Record) error {
	c.sawCancel = ctx.Err() != nil
	_, c.sawDeadline = ctx.Deadline()
	c.recs = append(c.recs, rec)
	return nil
}

type delayAudit struct {
	delay  time.Duration
	ctxErr error
}

func (d *delayAudit) Insert(ctx context.Context, _ speech.Record) error {
	select {
	case <-ctx.Done():
		d.ctxErr = ctx.Err()
		return ctx.Err()
	case <-time.After(d.delay):
		return nil
	}
}

type logSink struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (l *logSink) Write(p []byte) (int, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.buf.Write(p)
}

func (l *logSink) String() string {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.buf.String()
}
