package main

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
)

func TestQdrantCleanupRejectsUnverifiedBodies(t *testing.T) {
	cases := []struct {
		name        string
		delete      int
		body        string
		countStatus int
		count       string
		ok          bool
	}{
		{name: "completed and empty", delete: 200, body: `{"status":"ok","result":{"status":"completed"}}`, countStatus: 200, count: `{"status":"ok","result":{"count":0}}`, ok: true},
		{name: "completed but points remain", delete: 200, body: `{"status":"ok","result":{"status":"completed"}}`, countStatus: 200, count: `{"status":"ok","result":{"count":2}}`},
		{name: "substring in an error message", delete: 200, body: `{"status":"error","message":"operation completed already"}`, countStatus: 200, count: `{"status":"ok","result":{"count":0}}`},
		{name: "acknowledged", delete: 200, body: `{"status":"ok","result":{"status":"acknowledged"}}`, countStatus: 200, count: `{"status":"ok","result":{"count":0}}`},
		{name: "malformed", delete: 200, body: `{"status":`, countStatus: 200, count: `{"status":"ok","result":{"count":0}}`},
		{name: "http 500", delete: 500, body: `{"status":"ok","result":{"status":"completed"}}`, countStatus: 200, count: `{"status":"ok","result":{"count":0}}`},
		{name: "http 404", delete: 404, body: `{"status":"ok","result":{"status":"completed"}}`, countStatus: 200, count: `{"status":"ok","result":{"count":0}}`},
		{name: "count malformed", delete: 200, body: `{"status":"ok","result":{"status":"completed"}}`, countStatus: 200, count: `{"status":"ok"}`},
		{name: "count http error", delete: 200, body: `{"status":"ok","result":{"status":"completed"}}`, countStatus: 500, count: `{"status":"error","message":"completed"}`},
	}
	id := "11111111-1111-1111-1111-111111111111"
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var sawGeneration bool
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				raw, _ := io.ReadAll(r.Body)
				if !strings.Contains(string(raw), id) {
					t.Errorf("generation filter missing in %s", r.URL.Path)
				} else {
					sawGeneration = true
				}
				if !strings.Contains(r.URL.Path, "/collections/knowledge_chunks/") {
					w.WriteHeader(http.StatusNotFound)
					return
				}
				if strings.HasSuffix(r.URL.Path, "/points/count") {
					w.WriteHeader(tc.countStatus)
					_, _ = w.Write([]byte(tc.count))
					return
				}
				w.WriteHeader(tc.delete)
				_, _ = w.Write([]byte(tc.body))
			}))
			defer srv.Close()
			err := deleteQdrantVerified(srv.URL, "knowledge_chunks", id)
			if tc.ok && err != nil {
				t.Fatal(err)
			}
			if !tc.ok && err == nil {
				t.Fatal("accepted unverified delete")
			}
			if tc.ok && !sawGeneration {
				t.Fatal("count filter was not checked")
			}
		})
	}
}

func TestQdrantCleanupRejectsWrongCollection(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.Contains(r.URL.Path, "/collections/knowledge_chunks/") {
			w.WriteHeader(http.StatusNotFound)
			_, _ = w.Write([]byte(`{"status":"error"}`))
			return
		}
		_, _ = w.Write([]byte(`{"status":"ok","result":{"status":"completed","count":0}}`))
	}))
	defer srv.Close()
	if err := deleteQdrantVerified(srv.URL, "other_collection", "11111111-1111-1111-1111-111111111111"); err == nil {
		t.Fatal("wrong collection was treated as clean")
	}
}

func TestQdrantCleanupTimesOut(t *testing.T) {
	done := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		select {
		case <-done:
		case <-time.After(30 * time.Second):
		}
	}))
	defer func() {
		close(done)
		srv.Close()
	}()
	started := time.Now()
	if err := deleteQdrantVerified(srv.URL, "knowledge_chunks", "11111111-1111-1111-1111-111111111111"); err == nil {
		t.Fatal("timeout was ignored")
	}
	if time.Since(started) > 8*time.Second {
		t.Fatalf("delete waited %s", time.Since(started))
	}
}

func TestMissingQdrantConfigDoesNotClaimOrComplete(t *testing.T) {
	store := &fakeStore{}
	cases := []reaperConfig{
		{databaseURL: "postgres://cas@db/cas", collection: "knowledge_chunks"},
		{databaseURL: "postgres://cas@db/cas", qdrantURL: "http://qdrant:6333"},
		{qdrantURL: "http://qdrant:6333", collection: "knowledge_chunks"},
	}
	for _, cfg := range cases {
		store.claimCalled = false
		err := runReaper(context.Background(), cfg, store, func(string) error { return nil }, io.Discard)
		if err == nil || err.Error() != "config_missing" {
			t.Fatalf("config %v err %v", cfg, err)
		}
		if store.claimCalled || store.chunksDeleted || len(store.completed) != 0 {
			t.Fatalf("cleanup started without config: %+v", store)
		}
	}
	if err := deleteQdrantVerified("", "knowledge_chunks", "11111111-1111-1111-1111-111111111111"); err == nil {
		t.Fatal("empty QDRANT_URL succeeded")
	}
	if err := deleteQdrantVerified("http://qdrant", "", "11111111-1111-1111-1111-111111111111"); err == nil {
		t.Fatal("empty collection succeeded")
	}
}

func TestQdrantFailuresDoNotDeleteChunksOrComplete(t *testing.T) {
	id := "11111111-1111-1111-1111-111111111111"
	cfg := reaperConfig{databaseURL: "postgres://cas@db/cas", qdrantURL: "http://qdrant", collection: "knowledge_chunks", batch: 1}
	item := cleanupClaim{id: id, attempts: 1, token: uuid.New()}
	store := &fakeStore{claimed: []cleanupClaim{item}}
	var out bytes.Buffer
	err := runReaper(context.Background(), cfg, store, func(string) error { return errors.New("qdrant_failed") }, &out)
	if err == nil {
		t.Fatal("qdrant failure exited clean")
	}
	if store.chunksDeleted || len(store.completed) != 0 {
		t.Fatal("postgres chunks were deleted before qdrant verification")
	}
	if len(store.marked) != 1 || store.marked[0] != "qdrant_failed" {
		t.Fatalf("mark %v", store.marked)
	}
	text := out.String()
	if !strings.Contains(text, "cleanup_retryable_failed=1") || strings.Contains(text, "cleanup_completed=1") {
		t.Fatalf("metrics %s", text)
	}
	if strings.Contains(text, "qdrant_cleanup_failed") {
		t.Fatalf("generic qdrant line used for the failure: %s", text)
	}
}

func TestPostgresFailureAfterCleanQdrantDoesNotComplete(t *testing.T) {
	item := cleanupClaim{id: "11111111-1111-1111-1111-111111111111", attempts: 5, token: uuid.New()}
	store := &fakeStore{claimed: []cleanupClaim{item}, completeErr: errors.New("commit failed")}
	cfg := reaperConfig{databaseURL: "postgres://cas@db/cas", qdrantURL: "http://qdrant", collection: "knowledge_chunks"}
	var out bytes.Buffer
	err := runReaper(context.Background(), cfg, store, func(string) error { return nil }, &out)
	if err == nil {
		t.Fatal("postgres failure exited clean")
	}
	if store.chunksDeleted || len(store.completed) != 0 {
		t.Fatal("failed commit was recorded as completed")
	}
	if len(store.marked) != 1 || store.marked[0] != "postgres_failed" {
		t.Fatalf("mark %v", store.marked)
	}
	text := out.String()
	if !strings.Contains(text, "cleanup_terminal_failed=1") || !strings.Contains(text, "code=postgres_failed") {
		t.Fatalf("metrics %s", text)
	}
	if strings.Contains(text, "cleanup_completed=1") {
		t.Fatal(text)
	}
}

func TestLostClaimDoesNotDelete(t *testing.T) {
	item := cleanupClaim{id: "11111111-1111-1111-1111-111111111111", attempts: 1, token: uuid.New()}
	store := &fakeStore{claimed: []cleanupClaim{item}, renewErr: errCleanupLost}
	cfg := reaperConfig{databaseURL: "postgres://cas@db/cas", qdrantURL: "http://qdrant", collection: "knowledge_chunks"}
	var deleted bool
	var out bytes.Buffer
	err := runReaper(context.Background(), cfg, store, func(string) error {
		deleted = true
		return nil
	}, &out)
	if err == nil || deleted || store.chunksDeleted {
		t.Fatalf("lost claim deleted data err=%v deleted=%v", err, deleted)
	}
	if !strings.Contains(out.String(), "cleanup_lost=1") {
		t.Fatal(out.String())
	}
}

func TestClaimQueryIsBounded(t *testing.T) {
	if !strings.Contains(claimSQL, "LIMIT $5") || !strings.Contains(claimSQL, "FOR UPDATE OF g SKIP LOCKED") || !strings.Contains(claimSQL, "RETURNING") {
		t.Fatal("claim is not a bounded locked update")
	}
	if !strings.Contains(renewSQL, "cleanup_claim_token = $2") {
		t.Fatal("renew does not check the claim token")
	}
}

func TestReaperProcessExitsNonZeroWhenConfigMissing(t *testing.T) {
	if os.Getenv("CAS_REAPER_CHILD") == "1" {
		os.Unsetenv("DATABASE_URL")
		os.Unsetenv("QDRANT_URL")
		os.Unsetenv("QDRANT_COLLECTION")
		main()
		return
	}
	cmd := exec.Command(os.Args[0], "-test.run=TestReaperProcessExitsNonZeroWhenConfigMissing")
	cmd.Env = []string{"CAS_REAPER_CHILD=1"}
	err := cmd.Run()
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) || exitErr.ExitCode() == 0 {
		t.Fatalf("expected non-zero exit, got %v", err)
	}
}

type fakeStore struct {
	claimed       []cleanupClaim
	claimCalled   bool
	chunksDeleted bool
	completed     []string
	marked        []string
	completeErr   error
	renewErr      error
}

func (f *fakeStore) claim(context.Context, int, int) ([]cleanupClaim, error) {
	f.claimCalled = true
	return f.claimed, nil
}

func (f *fakeStore) renew(context.Context, cleanupClaim) error {
	return f.renewErr
}

func (f *fakeStore) complete(_ context.Context, item cleanupClaim) error {
	if f.completeErr != nil {
		return f.completeErr
	}
	f.chunksDeleted = true
	f.completed = append(f.completed, item.id)
	return nil
}

func (f *fakeStore) mark(_ context.Context, item cleanupClaim, code string) (string, error) {
	f.marked = append(f.marked, code)
	if item.attempts >= maxCleanupAttempts {
		return "terminal", nil
	}
	return "retryable", nil
}
