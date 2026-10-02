package middleware

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/gin-gonic/gin"
)

type memHandler struct {
	msgs []string
}

func (h *memHandler) Enabled(context.Context, slog.Level) bool { return true }

func (h *memHandler) Handle(_ context.Context, r slog.Record) error {
	var b strings.Builder
	b.WriteString(r.Message)
	r.Attrs(func(a slog.Attr) bool {
		b.WriteString(" ")
		b.WriteString(a.Key)
		b.WriteString("=")
		b.WriteString(a.Value.String())
		return true
	})
	h.msgs = append(h.msgs, b.String())
	return nil
}

func (h *memHandler) WithAttrs([]slog.Attr) slog.Handler { return h }
func (h *memHandler) WithGroup(string) slog.Handler      { return h }

func TestSensitivePath(t *testing.T) {
	if !SensitivePath("/api/v1/auth/login") || !SensitivePath("/api/v1/sessions/x/turns") {
		t.Fatal("expected sensitive")
	}
	if SensitivePath("/api/v1/procedures") {
		t.Fatal("procedures not sensitive")
	}
}

func TestRedactSecretsMasksTokens(t *testing.T) {
	in := `{"access_token":"jwt-secret-value","guest_token":"g-1","password":"secret-pass"}`
	out := RedactSecrets(in)
	for _, bad := range []string{"jwt-secret-value", "g-1", "secret-pass"} {
		if strings.Contains(out, bad) {
			t.Fatalf("leaked %q in %s", bad, out)
		}
	}
}

func TestRedactSecretsBearerAlreadyRedactedTerminates(t *testing.T) {
	// Regression: replacing Bearer tokens with "Bearer ***" must not re-match forever.
	in := `Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig next=ok`
	done := make(chan string, 1)
	go func() { done <- RedactSecrets(in) }()
	select {
	case out := <-done:
		if strings.Contains(out, "eyJhbGciOiJIUzI1NiJ9") {
			t.Fatalf("token leaked: %s", out)
		}
		if !strings.Contains(out, "Bearer ***") {
			t.Fatalf("expected redacted bearer: %s", out)
		}
		again := RedactSecrets(out)
		if again != out && !strings.Contains(again, "Bearer ***") {
			t.Fatalf("second pass unstable: %q -> %q", out, again)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("RedactSecrets hung on bearer masking")
	}
}

func TestApiLogDoesNotEmitBodiesForAuth(t *testing.T) {
	gin.SetMode(gin.TestMode)
	h := &memHandler{}
	r := gin.New()
	r.Use(RequestID(), ApiLog(slog.New(h)))
	r.POST("/api/v1/auth/login", func(c *gin.Context) {
		raw, _ := io.ReadAll(c.Request.Body)
		c.JSON(http.StatusOK, gin.H{"access_token": "super-secret-jwt", "echo_len": len(raw)})
	})

	body := `{"password":"hunter2","username":"admin"}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/auth/login", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	joined := strings.Join(h.msgs, "\n")
	for _, secret := range []string{"super-secret-jwt", "hunter2"} {
		if strings.Contains(joined, secret) {
			t.Fatalf("secret in logs: %s", joined)
		}
	}
	var resp map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if int(resp["echo_len"].(float64)) != len(body) {
		t.Fatalf("auth middleware consumed body: echo_len=%v want %d", resp["echo_len"], len(body))
	}
}

func TestSensitivePathDoesNotConsumeBody(t *testing.T) {
	gin.SetMode(gin.TestMode)
	h := &memHandler{}
	r := gin.New()
	r.Use(RequestID(), ApiLog(slog.New(h)))
	var got string
	r.POST("/api/v1/sessions/:id/turns", func(c *gin.Context) {
		raw, err := io.ReadAll(c.Request.Body)
		if err != nil {
			t.Errorf("read: %v", err)
		}
		got = string(raw)
		c.Status(http.StatusOK)
	})
	payload := `{"message":"` + strings.Repeat("ả", 500) + `"}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/sessions/00000000-0000-0000-0000-000000000001/turns",
		strings.NewReader(payload))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if got != payload {
		t.Fatalf("body mutated/truncated: got %d want %d runes=%d", len(got), len(payload), utf8.RuneCountInString(got))
	}
	joined := strings.Join(h.msgs, "\n")
	if strings.Contains(joined, "ảảả") {
		t.Fatal("sensitive session body leaked into logs")
	}
}

func TestNonSensitiveStreamsFullBodyBeyondLogPreview(t *testing.T) {
	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.Use(RequestID(), ApiLog(slog.Default()))
	var got string
	r.POST("/api/v1/procedures/echo", func(c *gin.Context) {
		raw, _ := io.ReadAll(c.Request.Body)
		got = string(raw)
		c.Status(http.StatusOK)
	})
	msg := strings.Repeat("Đăng ký khai sinh cho con tôi. ", 200)
	payload := `{"message":"` + msg + `"}`
	if len(payload) <= maxBodyLogBytes {
		t.Fatalf("fixture too small: %d", len(payload))
	}
	req := httptest.NewRequest(http.MethodPost, "/api/v1/procedures/echo", strings.NewReader(payload))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	_ = bytes.NewBuffer(nil)
	if got != payload {
		t.Fatalf("handler got truncated body: %d vs %d", len(got), len(payload))
	}
}

func TestNonSensitiveBodyLargerThanFormer2MiBCap(t *testing.T) {
	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.Use(RequestID(), ApiLog(slog.Default()))
	var gotLen int
	r.POST("/api/v1/procedures/echo", func(c *gin.Context) {
		raw, err := io.ReadAll(c.Request.Body)
		if err != nil {
			t.Errorf("handler read: %v", err)
			c.Status(http.StatusBadRequest)
			return
		}
		gotLen = len(raw)
		c.Status(http.StatusOK)
	})
	// > 2 MiB but under MaxRequestBodyBytes — must reach handler intact (no silent truncate).
	payload := strings.Repeat("A", (2<<20)+4096)
	req := httptest.NewRequest(http.MethodPost, "/api/v1/procedures/echo", strings.NewReader(payload))
	req.Header.Set("Content-Type", "text/plain")
	req.ContentLength = int64(len(payload))
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("want 200 got %d body=%s", w.Code, w.Body.String())
	}
	if gotLen != len(payload) {
		t.Fatalf("handler got %d want %d", gotLen, len(payload))
	}
}

func TestNonSensitiveBodyOverHardLimitReturns413(t *testing.T) {
	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.Use(RequestID(), ApiLog(slog.Default()))
	r.POST("/api/v1/procedures/echo", func(c *gin.Context) {
		t.Error("handler must not run for oversized Content-Length")
		c.Status(http.StatusOK)
	})
	req := httptest.NewRequest(http.MethodPost, "/api/v1/procedures/echo", strings.NewReader("x"))
	req.Header.Set("Content-Type", "text/plain")
	req.ContentLength = MaxRequestBodyBytes + 1
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("want 413 got %d", w.Code)
	}
}

// noLenReader refuses Len() so net/http leaves ContentLength=-1 (chunked-like).
type noLenReader struct{ r io.Reader }

func (n noLenReader) Read(p []byte) (int, error) { return n.r.Read(p) }

func TestChunkedBodyOverHardLimitReturns413(t *testing.T) {
	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.Use(RequestID(), ApiLog(slog.Default()))
	handlerRan := false
	r.POST("/api/v1/procedures/echo", func(c *gin.Context) {
		handlerRan = true
		_, _ = io.ReadAll(c.Request.Body)
		// Even if handler tries to write 400, guard must keep 413.
		c.JSON(http.StatusBadRequest, gin.H{"error": "should not win"})
	})
	// Real oversized body; ContentLength intentionally -1 (no fake header-only claim).
	payload := strings.Repeat("B", MaxRequestBodyBytes+2048)
	req := httptest.NewRequest(http.MethodPost, "/api/v1/procedures/echo", io.NopCloser(noLenReader{strings.NewReader(payload)}))
	req.Header.Set("Content-Type", "application/octet-stream")
	req.ContentLength = -1
	req.TransferEncoding = []string{"chunked"}
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("want 413 got %d body=%s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "PAYLOAD_TOO_LARGE") {
		t.Fatalf("expected PAYLOAD_TOO_LARGE in body: %s", w.Body.String())
	}
	_ = handlerRan // may or may not enter; status must still be 413
}

func TestSensitiveRouteChunkedBodyOverLimitReturns413WithoutLoggingBody(t *testing.T) {
	gin.SetMode(gin.TestMode)
	h := &memHandler{}
	r := gin.New()
	r.Use(RequestID(), ApiLog(slog.New(h)))
	r.POST("/api/v1/auth/login", func(c *gin.Context) {
		_, err := io.ReadAll(c.Request.Body)
		if err != nil {
			// Handler might see the limit error; must not override 413.
			c.JSON(http.StatusBadRequest, gin.H{"error": "bind failed"})
			return
		}
		c.JSON(http.StatusOK, gin.H{"ok": true})
	})
	secret := "hunter2-should-not-appear-in-logs"
	payload := secret + strings.Repeat("C", MaxRequestBodyBytes+1024)
	req := httptest.NewRequest(http.MethodPost, "/api/v1/auth/login", io.NopCloser(noLenReader{strings.NewReader(payload)}))
	req.Header.Set("Content-Type", "application/json")
	req.ContentLength = -1
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("want 413 got %d body=%s", w.Code, w.Body.String())
	}
	joined := strings.Join(h.msgs, "\n")
	if strings.Contains(joined, secret) {
		t.Fatalf("sensitive oversized body leaked into logs: %s", joined)
	}
	if !strings.Contains(joined, "[redacted]") {
		t.Fatalf("expected redacted markers in logs: %s", joined)
	}
}

func TestSensitiveSessionRouteEnforcesBodyLimit(t *testing.T) {
	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.Use(RequestID(), ApiLog(slog.Default()))
	r.POST("/api/v1/sessions/x/turns", func(c *gin.Context) {
		_, _ = io.ReadAll(c.Request.Body)
		c.JSON(http.StatusBadRequest, gin.H{"error": "nope"})
	})
	payload := strings.Repeat("D", MaxRequestBodyBytes+4096)
	req := httptest.NewRequest(http.MethodPost, "/api/v1/sessions/x/turns",
		io.NopCloser(noLenReader{strings.NewReader(payload)}))
	req.ContentLength = -1
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("want 413 got %d", w.Code)
	}
}

func TestDeleteWithOversizedChunkedBodyReturns413(t *testing.T) {
	// DELETE was previously skipped by method filter — must still enforce 32 MiB.
	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.Use(RequestID(), ApiLog(slog.Default()))
	r.DELETE("/api/v1/procedures/echo", func(c *gin.Context) {
		_, _ = io.ReadAll(c.Request.Body)
		c.JSON(http.StatusBadRequest, gin.H{"error": "should not win"})
	})
	payload := strings.Repeat("E", MaxRequestBodyBytes+1024)
	req := httptest.NewRequest(http.MethodDelete, "/api/v1/procedures/echo",
		io.NopCloser(noLenReader{strings.NewReader(payload)}))
	req.ContentLength = -1
	req.TransferEncoding = []string{"chunked"}
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("DELETE oversized body want 413 got %d body=%s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "PAYLOAD_TOO_LARGE") {
		t.Fatalf("expected PAYLOAD_TOO_LARGE: %s", w.Body.String())
	}
}

func TestGetWithOversizedDeclaredContentLengthReturns413(t *testing.T) {
	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.Use(RequestID(), ApiLog(slog.Default()))
	r.GET("/api/v1/procedures/echo", func(c *gin.Context) {
		t.Error("handler must not run for oversized GET body Content-Length")
		c.Status(http.StatusOK)
	})
	req := httptest.NewRequest(http.MethodGet, "/api/v1/procedures/echo", strings.NewReader("x"))
	req.ContentLength = MaxRequestBodyBytes + 1
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("GET oversized Content-Length want 413 got %d", w.Code)
	}
}
