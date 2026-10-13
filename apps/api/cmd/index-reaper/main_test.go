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
	"sync"
	"syscall"
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
		err := runReaper(context.Background(), cfg, store, func(context.Context, string) error { return nil }, io.Discard)
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
	err := runReaper(context.Background(), cfg, store, func(context.Context, string) error { return errors.New("qdrant_failed") }, &out)
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
	err := runReaper(context.Background(), cfg, store, func(context.Context, string) error { return nil }, &out)
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
	err := runReaper(context.Background(), cfg, store, func(context.Context, string) error {
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
	if !strings.Contains(claimSQL, "g.status IN ('STAGING', 'FAILED')") {
		t.Fatal("claim can pick a READY generation")
	}
	if !strings.Contains(claimSQL, "link.active_generation_id = g.id") {
		t.Fatal("claim can pick a generation that is still linked")
	}
	if !strings.Contains(claimSQL, "job.status = 'CLAIMED'") {
		t.Fatal("claim can pick a generation with a live job claim")
	}
}

func TestLoadConfigIntervalAndLoopStops(t *testing.T) {
	getenv := func(key string) string {
		switch key {
		case "DATABASE_URL":
			return "postgres://cas@db/cas"
		case "QDRANT_URL":
			return "http://qdrant"
		case "QDRANT_COLLECTION":
			return "knowledge_chunks"
		case "INDEX_REAPER_INTERVAL_SECONDS":
			return "1"
		default:
			return ""
		}
	}
	cfg, err := loadConfig(getenv)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.interval != time.Second {
		t.Fatalf("interval=%s", cfg.interval)
	}
	if cfg.shutdown != 30*time.Second {
		t.Fatalf("shutdown=%s", cfg.shutdown)
	}
	store := &fakeStore{}
	ctx, cancel := context.WithCancel(context.Background())
	var out bytes.Buffer
	done := make(chan error, 1)
	go func() {
		done <- runReaperLoop(ctx, cfg, store, func(context.Context, string) error { return nil }, &out)
	}()
	time.Sleep(20 * time.Millisecond)
	cancel()
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if !store.claimCalled {
		t.Fatal("loop did not run a cycle")
	}
	if !strings.Contains(out.String(), "reaper_stopped") {
		t.Fatalf("stop log missing: %s", out.String())
	}
}

func TestLoadConfigShutdownSeconds(t *testing.T) {
	getenv := func(key string) string {
		switch key {
		case "DATABASE_URL":
			return "postgres://cas@db/cas"
		case "QDRANT_URL":
			return "http://qdrant"
		case "QDRANT_COLLECTION":
			return "knowledge_chunks"
		case "INDEX_REAPER_SHUTDOWN_SECONDS":
			return "45"
		default:
			return ""
		}
	}
	cfg, err := loadConfig(getenv)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.shutdown != 45*time.Second {
		t.Fatalf("shutdown=%s", cfg.shutdown)
	}
	bad := func(key string) string {
		if key == "INDEX_REAPER_SHUTDOWN_SECONDS" {
			return "0"
		}
		return getenv(key)
	}
	if _, err := loadConfig(bad); err == nil {
		t.Fatal("zero shutdown was accepted")
	}
}

func TestIdleShutdownSkipsClaim(t *testing.T) {
	store := &fakeStore{claimed: []cleanupClaim{{id: "11111111-1111-1111-1111-111111111111", attempts: 1, token: uuid.New()}}}
	cfg := reaperConfig{
		databaseURL: "postgres://cas@db/cas",
		qdrantURL:   "http://qdrant",
		collection:  "knowledge_chunks",
		batch:       1,
		shutdown:    time.Second,
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := runOnce(ctx, cfg, store, func(context.Context, string) error {
		t.Fatal("idle shutdown started work")
		return nil
	}, io.Discard); err != nil {
		t.Fatal(err)
	}
	if store.claimCalled {
		t.Fatal("idle shutdown claimed work")
	}
}

func TestQdrantCleanupHonorsCanceledContext(t *testing.T) {
	started := make(chan struct{})
	var once sync.Once
	srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		once.Do(func() { close(started) })
		time.Sleep(2 * time.Second)
	}))
	defer srv.Close()
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		done <- deleteQdrantVerifiedCtx(ctx, srv.URL, "knowledge_chunks", "11111111-1111-1111-1111-111111111111")
	}()
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("request did not start")
	}
	cancel()
	select {
	case err := <-done:
		if err == nil {
			t.Fatal("canceled request succeeded")
		}
	case <-time.After(time.Second):
		t.Fatal("canceled request kept running")
	}
}

func TestShutdownDoesNotStartANewCycle(t *testing.T) {
	store := &countingStore{}
	cfg := reaperConfig{
		databaseURL: "postgres://cas@db/cas",
		qdrantURL:   "http://qdrant",
		collection:  "knowledge_chunks",
		batch:       1,
		interval:    20 * time.Millisecond,
		shutdown:    time.Second,
	}
	ctx, cancel := context.WithCancel(context.Background())
	var out bytes.Buffer
	done := make(chan error, 1)
	go func() {
		done <- runReaperLoop(ctx, cfg, store, func(context.Context, string) error { return nil }, &out)
	}()
	time.Sleep(50 * time.Millisecond)
	cancel()
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	after := store.claims
	time.Sleep(80 * time.Millisecond)
	if store.claims != after {
		t.Fatalf("new cycle after shutdown claims=%d after=%d", store.claims, after)
	}
	if !strings.Contains(out.String(), "reaper_shutdown_requested") || !strings.Contains(out.String(), "reaper_stopped") {
		t.Fatalf("logs %s", out.String())
	}
}

func TestInFlightWorkFinishesInsideShutdownGrace(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{})
	item := cleanupClaim{id: "11111111-1111-1111-1111-111111111111", attempts: 1, token: uuid.New()}
	store := &fakeStore{claimed: []cleanupClaim{item}}
	cfg := reaperConfig{
		databaseURL: "postgres://cas@db/cas",
		qdrantURL:   "http://qdrant",
		collection:  "knowledge_chunks",
		batch:       1,
		shutdown:    time.Second,
	}
	parent, cancel := context.WithCancel(context.Background())
	var out bytes.Buffer
	done := make(chan error, 1)
	go func() {
		done <- runOnce(parent, cfg, store, func(context.Context, string) error {
			close(started)
			<-release
			return nil
		}, &out)
	}()
	<-started
	cancel()
	time.Sleep(20 * time.Millisecond)
	close(release)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if len(store.completed) != 1 {
		t.Fatalf("in-flight work was dropped: %+v %s", store, out.String())
	}
	if !strings.Contains(out.String(), "reaper_draining") {
		t.Fatalf("drain log missing: %s", out.String())
	}
}

func TestShutdownTimeoutCancelsInFlightWork(t *testing.T) {
	started := make(chan struct{})
	item := cleanupClaim{id: "11111111-1111-1111-1111-111111111111", attempts: 1, token: uuid.New()}
	store := &fakeStore{claimed: []cleanupClaim{item}}
	cfg := reaperConfig{
		databaseURL: "postgres://cas@db/cas",
		qdrantURL:   "http://qdrant",
		collection:  "knowledge_chunks",
		batch:       1,
		shutdown:    30 * time.Millisecond,
	}
	parent, cancel := context.WithCancel(context.Background())
	var out bytes.Buffer
	done := make(chan error, 1)
	go func() {
		done <- runOnce(parent, cfg, store, func(ctx context.Context, _ string) error {
			close(started)
			<-ctx.Done()
			return ctx.Err()
		}, &out)
	}()
	<-started
	cancel()
	if err := <-done; err == nil {
		t.Fatal("timeout did not surface")
	}
	if !strings.Contains(out.String(), "reaper_shutdown_timeout") {
		t.Fatalf("timeout log missing: %s", out.String())
	}
	if len(store.completed) != 0 {
		t.Fatal("incomplete delete was marked completed")
	}
}

func TestRepeatedShutdownDoesNotPanic(t *testing.T) {
	cfg := reaperConfig{
		databaseURL: "postgres://cas@db/cas",
		qdrantURL:   "http://qdrant",
		collection:  "knowledge_chunks",
		interval:    time.Hour,
		shutdown:    time.Second,
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	cancel()
	var out bytes.Buffer
	if err := runReaperLoop(ctx, cfg, &fakeStore{}, func(context.Context, string) error { return nil }, &out); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "reaper_stopped") {
		t.Fatal(out.String())
	}
}

func TestNotifySignalsExitOnSIGTERM(t *testing.T) {
	if os.Getenv("CAS_REAPER_CHILD") == "sigterm" {
		ctx, stop := notifySignals(context.Background())
		defer stop()
		<-ctx.Done()
		os.Exit(0)
	}
	cmd := exec.Command(os.Args[0], "-test.run=TestNotifySignalsExitOnSIGTERM")
	cmd.Env = append(os.Environ(), "CAS_REAPER_CHILD=sigterm")
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	time.Sleep(50 * time.Millisecond)
	if err := cmd.Process.Signal(syscall.SIGTERM); err != nil {
		t.Fatal(err)
	}
	if err := cmd.Wait(); err != nil {
		t.Fatalf("child %v", err)
	}
}

func TestNotifySignalsExitOnSIGINT(t *testing.T) {
	if os.Getenv("CAS_REAPER_CHILD") == "sigint" {
		ctx, stop := notifySignals(context.Background())
		defer stop()
		<-ctx.Done()
		os.Exit(0)
	}
	cmd := exec.Command(os.Args[0], "-test.run=TestNotifySignalsExitOnSIGINT")
	cmd.Env = append(os.Environ(), "CAS_REAPER_CHILD=sigint")
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	time.Sleep(50 * time.Millisecond)
	if err := cmd.Process.Signal(os.Interrupt); err != nil {
		t.Fatal(err)
	}
	if err := cmd.Wait(); err != nil {
		t.Fatalf("child %v", err)
	}
}

type countingStore struct {
	claims int
}

func (c *countingStore) claim(context.Context, int, int) ([]cleanupClaim, error) {
	c.claims++
	return nil, nil
}

func (c *countingStore) renew(context.Context, cleanupClaim) error { return nil }

func (c *countingStore) complete(context.Context, cleanupClaim) error { return nil }

func (c *countingStore) mark(context.Context, cleanupClaim, string) (string, error) {
	return "retryable", nil
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
