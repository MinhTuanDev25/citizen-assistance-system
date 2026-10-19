package retrieve

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestClientPropagatesAuthAndRequestID(t *testing.T) {
	var gotAuth, gotRID, gotPath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		gotRID = r.Header.Get("X-Request-ID")
		gotPath = r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(Response{SchemaVersion: SchemaVersion, Status: "no_relevant_evidence", Hits: []Hit{}})
	}))
	defer srv.Close()
	c := NewClient(srv.URL, time.Second)
	c.ServiceToken = "local-dev-ai-service-token"
	_, err := c.Retrieve(context.Background(), Request{
		Query: "khai sinh", XaID: "xa_chu_se", ProcedureVersionID: "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
		RequestID: "11111111-1111-1111-1111-111111111111", AllowedSources: []Source{},
	})
	if err != nil {
		t.Fatal(err)
	}
	if gotPath != "/v1/retrieve" {
		t.Fatalf("path %s", gotPath)
	}
	if gotAuth != "Bearer local-dev-ai-service-token" {
		t.Fatalf("auth %q", gotAuth)
	}
	if gotRID != "11111111-1111-1111-1111-111111111111" {
		t.Fatalf("rid %q", gotRID)
	}
}

func TestClientRejectsUnknownFieldsAndBadStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"schema_version":"retrieve.v1","status":"ok","hits":[],"extra":true}`)
	}))
	defer srv.Close()
	c := NewClient(srv.URL, time.Second)
	if _, err := c.Retrieve(context.Background(), Request{Query: "x", XaID: "xa_chu_se", ProcedureVersionID: "v"}); err == nil {
		t.Fatal("unknown field must fail")
	}
}

func TestClientMapsNonOKToUnavailable(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "down", http.StatusServiceUnavailable)
	}))
	defer srv.Close()
	c := NewClient(srv.URL, time.Second)
	_, err := c.Retrieve(context.Background(), Request{Query: "x", XaID: "xa", ProcedureVersionID: "v"})
	if err == nil || !strings.Contains(err.Error(), "unavailable") {
		t.Fatalf("err=%v", err)
	}
}

func TestClientPreservesFastAPIUnavailableReason(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = io.WriteString(w, `{"detail":{"reason":"embedding_unavailable"}}`)
	}))
	defer srv.Close()
	c := NewClient(srv.URL, time.Second)
	_, err := c.Retrieve(context.Background(), Request{Query: "x", XaID: "xa", ProcedureVersionID: "v"})
	if UnavailableReason(err) != "embedding_unavailable" {
		t.Fatalf("reason=%q err=%v", UnavailableReason(err), err)
	}
	if !errors.Is(err, ErrUnavailable) {
		t.Fatalf("must unwrap ErrUnavailable: %v", err)
	}
}

func TestPassesRelatedFloor(t *testing.T) {
	low := 0.01
	zero := 0.0
	neg := -0.8
	ok := 0.77
	if PassesRelatedFloor(nil) || PassesRelatedFloor(&low) || PassesRelatedFloor(&zero) || PassesRelatedFloor(&neg) {
		t.Fatal("noise scores must fail the related floor")
	}
	if !PassesRelatedFloor(&ok) {
		t.Fatal("0.77 must pass the related floor")
	}
}

func TestClientTimeoutIsBounded(t *testing.T) {
	block := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-block
	}))
	defer func() {
		close(block)
		srv.Close()
	}()
	c := NewClient(srv.URL, 30*time.Millisecond)
	start := time.Now()
	_, err := c.Retrieve(context.Background(), Request{Query: "x", XaID: "xa", ProcedureVersionID: "v"})
	if err == nil {
		t.Fatal("expected timeout")
	}
	if time.Since(start) > 2*time.Second {
		t.Fatalf("timeout not bounded")
	}
}

func TestClientAcceptsIncompatibleStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(Response{
			SchemaVersion: SchemaVersion,
			Status:        "retrieval_incompatible",
			Reason:        "embedding_identity_mismatch",
			Hits:          []Hit{},
		})
	}))
	defer srv.Close()
	c := NewClient(srv.URL, time.Second)
	resp, err := c.Retrieve(context.Background(), Request{Query: "x", XaID: "xa", ProcedureVersionID: "v"})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Status != "retrieval_incompatible" || resp.Reason != "embedding_identity_mismatch" {
		t.Fatalf("%#v", resp)
	}
}

func TestNewClientClampsTimeout(t *testing.T) {
	c := NewClient("http://example.invalid", 999*time.Second)
	if c.Timeout != HardMaxTimeout {
		t.Fatalf("got %s", c.Timeout)
	}
}
