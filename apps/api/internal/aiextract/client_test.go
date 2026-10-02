package aiextract

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestClientPropagatesRequestIDHeader(t *testing.T) {
	var gotHeader string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotHeader = r.Header.Get("X-Request-ID")
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(Response{SchemaVersion: SchemaVersion, Provider: "mock", Model: "mock"})
	}))
	defer srv.Close()

	c := NewClient(srv.URL, time.Second)
	req := Request{SchemaVersion: SchemaVersion, RequestID: "11111111-1111-1111-1111-111111111111", Message: "x"}
	if _, err := c.Extract(context.Background(), req); err != nil {
		t.Fatalf("Extract: %v", err)
	}
	if gotHeader != req.RequestID {
		t.Fatalf("X-Request-ID = %q, want %q", gotHeader, req.RequestID)
	}
}

func TestClientTimeoutIsBoundedAndReported(t *testing.T) {
	block := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-block
	}))
	// Order matters: close(block) MUST run before srv.Close(), otherwise
	// Close() deadlocks waiting for the still-blocked in-flight handler.
	defer func() {
		close(block)
		srv.Close()
	}()

	c := NewClient(srv.URL, 30*time.Millisecond)
	start := time.Now()
	_, err := c.Extract(context.Background(), Request{SchemaVersion: SchemaVersion, RequestID: "r", Message: "x"})
	elapsed := time.Since(start)
	if err == nil {
		t.Fatal("expected timeout error, got nil")
	}
	if elapsed > 2*time.Second {
		t.Fatalf("client did not respect timeout, took %s", elapsed)
	}
}

func TestClientTimeoutClampedToHardMax(t *testing.T) {
	c := NewClient("http://example.invalid", 999*time.Second)
	if c.Timeout != HardMaxTimeout {
		t.Fatalf("Timeout = %s, want clamp to %s", c.Timeout, HardMaxTimeout)
	}
	c2 := NewClient("http://example.invalid", 0)
	if c2.Timeout != HardMaxTimeout {
		t.Fatalf("zero timeout should clamp to hard max, got %s", c2.Timeout)
	}
}

func TestClientRejectsWrongSchemaVersion(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"schema_version": "extract.v2", "provider": "mock", "model": "mock"})
	}))
	defer srv.Close()

	c := NewClient(srv.URL, time.Second)
	_, err := c.Extract(context.Background(), Request{SchemaVersion: SchemaVersion, RequestID: "r", Message: "x"})
	if err == nil {
		t.Fatal("expected error for wrong schema_version")
	}
}

func TestClientRejectsNon200Status(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
	}))
	defer srv.Close()

	c := NewClient(srv.URL, time.Second)
	_, err := c.Extract(context.Background(), Request{SchemaVersion: SchemaVersion, RequestID: "r", Message: "x"})
	if err == nil {
		t.Fatal("expected error for non-200 status")
	}
}

func TestClientRejectsOversizedResponse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		huge := strings.Repeat("x", int(DefaultMaxResponseBytes)+1024)
		_, _ = w.Write([]byte(`{"schema_version":"extract.v1","provider":"mock","model":"mock","padding":"` + huge + `"}`))
	}))
	defer srv.Close()

	c := NewClient(srv.URL, time.Second)
	_, err := c.Extract(context.Background(), Request{SchemaVersion: SchemaVersion, RequestID: "r", Message: "x"})
	if err == nil {
		t.Fatal("expected error for oversized response")
	}
}

func TestClientRejectsUnknownField(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"schema_version":"extract.v1","provider":"mock","model":"mock","extra":true}`))
	}))
	defer srv.Close()
	c := NewClient(srv.URL, time.Second)
	_, err := c.Extract(context.Background(), Request{SchemaVersion: SchemaVersion, RequestID: "r", Message: "x"})
	if !errors.Is(err, ErrInvalidResponse) {
		t.Fatalf("got %v, want ErrInvalidResponse", err)
	}
}

func TestClientRejectsTrailingJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"schema_version":"extract.v1","provider":"mock","model":"mock"}{"x":1}`))
	}))
	defer srv.Close()
	c := NewClient(srv.URL, time.Second)
	_, err := c.Extract(context.Background(), Request{SchemaVersion: SchemaVersion, RequestID: "r", Message: "x"})
	if !errors.Is(err, ErrInvalidResponse) {
		t.Fatalf("got %v, want ErrInvalidResponse", err)
	}
}

func TestClientRateLimitIsSingleAttempt(t *testing.T) {
	var calls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		w.Header().Set("Retry-After", "3")
		w.WriteHeader(http.StatusTooManyRequests)
	}))
	defer srv.Close()
	c := NewClient(srv.URL, time.Second)
	_, err := c.Extract(context.Background(), Request{SchemaVersion: SchemaVersion, RequestID: "r", Message: "x"})
	if !errors.Is(err, ErrRateLimited) {
		t.Fatalf("got %v, want ErrRateLimited", err)
	}
	if calls != 1 {
		t.Fatalf("calls = %d, want 1 (no retry)", calls)
	}
	if strings.Contains(err.Error(), "Retry-After") {
		t.Fatalf("error included response header: %v", err)
	}
}

func TestClientSendsServiceTokenAndDoesNotEchoIt(t *testing.T) {
	const token = "svc-token-not-for-logs"
	var got string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = r.Header.Get("Authorization")
		w.WriteHeader(http.StatusUnauthorized)
	}))
	defer srv.Close()
	c := NewClient(srv.URL, time.Second)
	c.ServiceToken = token
	_, err := c.Extract(context.Background(), Request{SchemaVersion: SchemaVersion, RequestID: "r", Message: "x"})
	if err == nil {
		t.Fatal("expected non-200")
	}
	if got != "Bearer "+token {
		t.Fatalf("Authorization = %q", got)
	}
	if strings.Contains(err.Error(), token) {
		t.Fatalf("error echoed the service token: %v", err)
	}
}

func TestDisabledClientReturnsErrDisabled(t *testing.T) {
	var c *Client
	if _, err := c.Extract(context.Background(), Request{}); err != ErrDisabled {
		t.Fatalf("nil client: got err=%v, want ErrDisabled", err)
	}
	c2 := &Client{}
	if _, err := c2.Extract(context.Background(), Request{}); err != ErrDisabled {
		t.Fatalf("empty BaseURL: got err=%v, want ErrDisabled", err)
	}
}
