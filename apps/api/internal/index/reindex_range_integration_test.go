//go:build integration

package index_test

import (
	"context"
	"net/http"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/index"
)

func TestReindexPageRangeAtomicAndIdempotent(t *testing.T) {
	pool := requireDB(t)
	versionID := insertVersion(t, pool, "xa_chu_se", "ho_tich_chung_thuc", "p5_range_"+uuid.NewString()[:8])
	t.Cleanup(func() { cleanupVersion(pool, versionID) })
	var fail atomic.Bool
	var calls atomic.Int32
	var mu sync.Mutex
	var seen []string
	worker := workerFn(func(_ context.Context, req index.Request) (index.Response, error) {
		calls.Add(1)
		mu.Lock()
		if req.PageRange == nil {
			seen = append(seen, "")
		} else {
			seen = append(seen, *req.PageRange)
		}
		mu.Unlock()
		if fail.Load() {
			code := "worker_failed"
			return index.Response{
				SchemaVersion: index.SchemaVersion, DocumentID: req.DocumentID,
				ProcedureVersionID: req.ProcedureVersionID, Outcome: index.OutcomeFailed, ErrorCode: &code,
			}, nil
		}
		return readyResponse(req), nil
	})
	eng, admin := engine(t, pool, worker, true)
	docID := uploadDoc(t, eng, admin, "%PDF-1.4 reindex-range")
	t.Cleanup(func() { cleanupDoc(pool, docID) })
	res := call(t, eng, http.MethodPost, "/api/v1/admin/documents/"+docID+"/links", admin, uuid.New(), map[string]string{
		"procedure_version_id": versionID, "relationship_type": "SOURCE", "page_range": "1-2",
	})
	if res.Code != http.StatusCreated {
		t.Fatalf("link %d %s", res.Code, res.Body.String())
	}
	indexed := call(t, eng, http.MethodPost, indexPath(docID, versionID, false), admin, uuid.New(), nil)
	if indexed.Code != http.StatusOK {
		t.Fatalf("index %d %s", indexed.Code, indexed.Body.String())
	}
	if got := linkPageRange(t, pool, docID, versionID); got != "1-2" {
		t.Fatalf("after index page_range=%q", got)
	}
	if got := latestJobPageRange(t, pool, docID); got != "1-2" {
		t.Fatalf("first job snapshot=%q", got)
	}

	fail.Store(false)
	reqID := uuid.New()
	ok := call(t, eng, http.MethodPost, "/api/v1/admin/documents/"+docID+"/links/"+versionID+"/index/reindex", admin, reqID, map[string]string{
		"page_range": "10-21",
	})
	if ok.Code != http.StatusOK {
		t.Fatalf("reindex set %d %s", ok.Code, ok.Body.String())
	}
	if got := linkPageRange(t, pool, docID, versionID); got != "10-21" {
		t.Fatalf("published page_range=%q", got)
	}
	beforeReplay := calls.Load()
	replay := call(t, eng, http.MethodPost, "/api/v1/admin/documents/"+docID+"/links/"+versionID+"/index/reindex", admin, reqID, map[string]string{
		"page_range": "10-21",
	})
	if replay.Code != http.StatusOK {
		t.Fatalf("replay %d %s", replay.Code, replay.Body.String())
	}
	if calls.Load() != beforeReplay {
		t.Fatalf("replay ran worker %d -> %d", beforeReplay, calls.Load())
	}
	conflict := call(t, eng, http.MethodPost, "/api/v1/admin/documents/"+docID+"/links/"+versionID+"/index/reindex", admin, reqID, map[string]string{
		"page_range": "5-6",
	})
	if conflict.Code != http.StatusConflict {
		t.Fatalf("range conflict %d %s", conflict.Code, conflict.Body.String())
	}
	if got := linkPageRange(t, pool, docID, versionID); got != "10-21" {
		t.Fatalf("conflict mutated page_range=%q", got)
	}

	keep := call(t, eng, http.MethodPost, "/api/v1/admin/documents/"+docID+"/links/"+versionID+"/index/reindex", admin, uuid.New(), nil)
	if keep.Code != http.StatusOK {
		t.Fatalf("keep %d %s", keep.Code, keep.Body.String())
	}
	if got := linkPageRange(t, pool, docID, versionID); got != "10-21" {
		t.Fatalf("keep mutated page_range=%q", got)
	}
	clear := call(t, eng, http.MethodPost, "/api/v1/admin/documents/"+docID+"/links/"+versionID+"/index/reindex", admin, uuid.New(), map[string]string{
		"page_range": "",
	})
	if clear.Code != http.StatusOK {
		t.Fatalf("clear %d %s", clear.Code, clear.Body.String())
	}
	if got := linkPageRange(t, pool, docID, versionID); got != "" {
		t.Fatalf("clear left page_range=%q", got)
	}

	fail.Store(true)
	failed := call(t, eng, http.MethodPost, "/api/v1/admin/documents/"+docID+"/links/"+versionID+"/index/reindex", admin, uuid.New(), map[string]string{
		"page_range": "5-6",
	})
	if failed.Code != http.StatusOK {
		t.Fatalf("failed reindex %d %s", failed.Code, failed.Body.String())
	}
	mu.Lock()
	if len(seen) == 0 || seen[len(seen)-1] != "5-6" {
		t.Fatalf("worker did not see job range: %v", seen)
	}
	mu.Unlock()
	if got := linkPageRange(t, pool, docID, versionID); got != "" {
		t.Fatalf("failed reindex mutated page_range=%q", got)
	}
}

func TestConcurrentReindexDoesNotMutateLinkBeforeClaim(t *testing.T) {
	pool := requireDB(t)
	versionID := insertVersion(t, pool, "xa_chu_se", "ho_tich_chung_thuc", "p5_range_c_"+uuid.NewString()[:8])
	t.Cleanup(func() { cleanupVersion(pool, versionID) })
	started := make(chan struct{})
	release := make(chan struct{})
	var firstRange atomic.Value
	worker := workerFn(func(_ context.Context, req index.Request) (index.Response, error) {
		if req.PageRange != nil && *req.PageRange != "1-2" {
			firstRange.Store(*req.PageRange)
			select {
			case <-started:
			default:
				close(started)
				<-release
			}
		}
		return readyResponse(req), nil
	})
	eng, admin := engine(t, pool, worker, true)
	docID := uploadDoc(t, eng, admin, "%PDF-1.4 concurrent-reindex")
	t.Cleanup(func() { cleanupDoc(pool, docID) })
	linked := call(t, eng, http.MethodPost, "/api/v1/admin/documents/"+docID+"/links", admin, uuid.New(), map[string]string{
		"procedure_version_id": versionID, "relationship_type": "SOURCE", "page_range": "1-2",
	})
	if linked.Code != http.StatusCreated {
		t.Fatalf("link %d %s", linked.Code, linked.Body.String())
	}
	if call(t, eng, http.MethodPost, indexPath(docID, versionID, false), admin, uuid.New(), nil).Code != http.StatusOK {
		t.Fatal("index")
	}
	var wg sync.WaitGroup
	codes := make(chan int, 2)
	wg.Add(2)
	go func() {
		defer wg.Done()
		codes <- call(t, eng, http.MethodPost, "/api/v1/admin/documents/"+docID+"/links/"+versionID+"/index/reindex", admin, uuid.New(), map[string]string{
			"page_range": "10-21",
		}).Code
	}()
	select {
	case <-started:
	case <-time.After(5 * time.Second):
		t.Fatal("first reindex did not start")
	}
	if got := linkPageRange(t, pool, docID, versionID); got != "1-2" {
		t.Fatalf("in-flight reindex mutated page_range=%q", got)
	}
	go func() {
		defer wg.Done()
		codes <- call(t, eng, http.MethodPost, "/api/v1/admin/documents/"+docID+"/links/"+versionID+"/index/reindex", admin, uuid.New(), map[string]string{
			"page_range": "5-6",
		}).Code
	}()
	second := <-codes
	if second != http.StatusConflict {
		t.Fatalf("second reindex %d", second)
	}
	if got := linkPageRange(t, pool, docID, versionID); got != "1-2" {
		t.Fatalf("conflict mutated page_range=%q", got)
	}
	close(release)
	wg.Wait()
	close(codes)
	first := <-codes
	if first != http.StatusOK {
		t.Fatalf("first reindex %d", first)
	}
	if got := linkPageRange(t, pool, docID, versionID); got != "10-21" {
		t.Fatalf("published page_range=%q want 10-21 (worker saw %v)", got, firstRange.Load())
	}
}

func linkPageRange(t *testing.T, pool *pgxpool.Pool, docID, versionID string) string {
	t.Helper()
	var page *string
	if err := pool.QueryRow(context.Background(), `
		SELECT page_range FROM procedure_version_documents
		WHERE document_id = $1 AND procedure_version_id = $2`, docID, versionID).Scan(&page); err != nil {
		t.Fatal(err)
	}
	if page == nil {
		return ""
	}
	return *page
}

func latestJobPageRange(t *testing.T, pool *pgxpool.Pool, docID string) string {
	t.Helper()
	var page *string
	if err := pool.QueryRow(context.Background(), `
		SELECT page_range FROM document_index_jobs
		WHERE document_id = $1
		ORDER BY created_at DESC
		LIMIT 1`, docID).Scan(&page); err != nil {
		t.Fatal(err)
	}
	if page == nil {
		return ""
	}
	return *page
}

func activeGeneration(t *testing.T, pool *pgxpool.Pool, docID, versionID string) string {
	t.Helper()
	var gen *string
	if err := pool.QueryRow(context.Background(), `
		SELECT active_generation_id::text FROM procedure_version_documents
		WHERE document_id = $1 AND procedure_version_id = $2`, docID, versionID).Scan(&gen); err != nil {
		t.Fatal(err)
	}
	if gen == nil {
		return ""
	}
	return *gen
}
