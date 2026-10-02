//go:build integration

package index_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/auth"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/config"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/httpserver"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/index"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/storage"
)

func TestPipelinePublishKeepsLastGoodAndRejectsLateToken(t *testing.T) {
	pool := requireDB(t)
	defer pool.Close()
	versionID := oneVersion(t, pool, "xa_chu_se", "ho_tich_chung_thuc")
	var procedureID string
	if err := pool.QueryRow(context.Background(), `SELECT procedure_id FROM procedure_versions WHERE id = $1`, versionID).Scan(&procedureID); err != nil {
		t.Fatal(err)
	}
	worker := &scriptWorker{pool: pool, procedureID: procedureID, mode: "fail"}
	eng, admin := pipelineEngine(t, pool, worker)
	docID := uploadDoc(t, eng, admin, "%PDF-1.4\n%pipeline-"+uuid.NewString()+"\n")
	t.Cleanup(func() { cleanupDoc(pool, docID) })
	linkDoc(t, eng, admin, docID, versionID)
	failed := call(t, eng, http.MethodPost, "/api/v1/admin/documents/"+docID+"/links/"+versionID+"/index", admin, uuid.New(), nil)
	if failed.Code != http.StatusOK || !bytesContains(failed.Body.String(), `"link_status":"FAILED"`) {
		t.Fatalf("fail %d %s", failed.Code, failed.Body.String())
	}
	worker.mode = "ok"
	readyRes := call(t, eng, http.MethodPost, "/api/v1/admin/documents/"+docID+"/links/"+versionID+"/index/retry", admin, uuid.New(), nil)
	if readyRes.Code != http.StatusOK || !bytesContains(readyRes.Body.String(), `"link_status":"READY"`) {
		t.Fatalf("publish %d %s", readyRes.Code, readyRes.Body.String())
	}
	var ready int
	if err := pool.QueryRow(context.Background(), `
		SELECT count(*) FROM document_index_generations
		WHERE document_id = $1 AND status = 'READY'`, docID).Scan(&ready); err != nil {
		t.Fatal(err)
	}
	if ready != 1 {
		t.Fatalf("ready generations %d", ready)
	}
	var active string
	if err := pool.QueryRow(context.Background(), `
		SELECT active_generation_id::text FROM procedure_version_documents
		WHERE document_id = $1 AND procedure_version_id = $2`, docID, versionID).Scan(&active); err != nil {
		t.Fatal(err)
	}
	worker.mode = "mismatch"
	again := call(t, eng, http.MethodPost, "/api/v1/admin/documents/"+docID+"/links/"+versionID+"/index/retry", admin, uuid.New(), nil)
	if again.Code != http.StatusUnprocessableEntity {
		// READY cannot be retried through the P4A state machine. A bad generation must not replace the active one.
		t.Fatalf("reindex from READY %d %s", again.Code, again.Body.String())
	}
	var still string
	if err := pool.QueryRow(context.Background(), `
		SELECT active_generation_id::text FROM procedure_version_documents
		WHERE document_id = $1 AND procedure_version_id = $2`, docID, versionID).Scan(&still); err != nil {
		t.Fatal(err)
	}
	if still != active {
		t.Fatalf("active generation changed %s -> %s", active, still)
	}
	worker.mode = "late"
	late := call(t, eng, http.MethodPost, "/api/v1/admin/documents/"+docID+"/links/"+versionID+"/index/reindex", admin, uuid.New(), nil)
	if late.Code != http.StatusConflict {
		t.Fatalf("late token %d %s", late.Code, late.Body.String())
	}
	if err := pool.QueryRow(context.Background(), `
		SELECT active_generation_id::text FROM procedure_version_documents
		WHERE document_id = $1 AND procedure_version_id = $2`, docID, versionID).Scan(&still); err != nil {
		t.Fatal(err)
	}
	if still != active {
		t.Fatalf("late publish changed active %s -> %s", active, still)
	}
	if _, err := pool.Exec(context.Background(), `
		UPDATE document_index_jobs
		SET claim_expires_at = now() - interval '1 second'
		WHERE document_id = $1 AND status = 'CLAIMED'`, docID); err != nil {
		t.Fatal(err)
	}
	worker.mode = "fail"
	failedReindex := call(t, eng, http.MethodPost, "/api/v1/admin/documents/"+docID+"/links/"+versionID+"/index/reindex", admin, uuid.New(), nil)
	if failedReindex.Code != http.StatusOK || !bytesContains(failedReindex.Body.String(), `"link_status":"READY"`) {
		t.Fatalf("failed reindex %d %s", failedReindex.Code, failedReindex.Body.String())
	}
	var reindexErr *string
	if err := pool.QueryRow(context.Background(), `
		SELECT active_generation_id::text, reindex_error_code FROM procedure_version_documents
		WHERE document_id = $1 AND procedure_version_id = $2`, docID, versionID).Scan(&still, &reindexErr); err != nil {
		t.Fatal(err)
	}
	if still != active || reindexErr == nil || *reindexErr == "" {
		t.Fatalf("last known good lost active=%s err=%v", still, reindexErr)
	}
	worker.mode = "ok"
	againReady := call(t, eng, http.MethodPost, "/api/v1/admin/documents/"+docID+"/links/"+versionID+"/index/reindex", admin, uuid.New(), nil)
	if againReady.Code != http.StatusOK || !bytesContains(againReady.Body.String(), `"link_status":"READY"`) {
		t.Fatalf("reindex %d %s", againReady.Code, againReady.Body.String())
	}
	var superseded int
	if err := pool.QueryRow(context.Background(), `
		SELECT count(*) FILTER (WHERE status = 'READY'), count(*) FILTER (WHERE status = 'SUPERSEDED')
		FROM document_index_generations WHERE document_id = $1`, docID).Scan(&ready, &superseded); err != nil {
		t.Fatal(err)
	}
	if ready != 1 || superseded < 1 {
		t.Fatalf("ready %d superseded %d", ready, superseded)
	}
	if err := pool.QueryRow(context.Background(), `
		SELECT active_generation_id::text FROM procedure_version_documents
		WHERE document_id = $1 AND procedure_version_id = $2`, docID, versionID).Scan(&still); err != nil {
		t.Fatal(err)
	}
	if still == active {
		t.Fatal("successful reindex did not move active generation")
	}
}

func bytesContains(body, needle string) bool {
	return len(body) >= len(needle) && (body == needle || len(needle) > 0 && stringIndex(body, needle))
}

func stringIndex(body, needle string) bool {
	for i := 0; i+len(needle) <= len(body); i++ {
		if body[i:i+len(needle)] == needle {
			return true
		}
	}
	return false
}

type scriptWorker struct {
	pool        *pgxpool.Pool
	procedureID string
	mode        string
	decoy       uuid.UUID
	entered     chan struct{}
	release     chan struct{}
}

func (w *scriptWorker) Index(ctx context.Context, req index.Request) (index.Response, error) {
	if w.entered != nil {
		close(w.entered)
		w.entered = nil
	}
	if w.release != nil {
		<-w.release
	}
	if req.SchemaVersion != index.SchemaVersionV2 || req.GenerationID == uuid.Nil {
		return index.Response{}, context.Canceled
	}
	if w.mode == "wrong" {
		decoy := uuid.New()
		_, err := w.pool.Exec(ctx, `
			INSERT INTO document_index_generations (
				id, xa_id, document_id, procedure_version_id, job_id, status,
				pipeline_version, extraction_version, ocr_version, chunk_config_hash,
				embedding_model_id, embedding_revision, embedding_checksum, vector_dimension,
				source_sha256, content_sha256, manifest_hash
			) VALUES ($1,$2,$3,$4,$5,'STAGING','pending','pending','pending','pending','pending','pending','',0,$6,'','')`,
			decoy, req.XAID, req.DocumentID, req.ProcedureVersionID, req.JobID, req.Checksum)
		if err != nil {
			return index.Response{}, err
		}
		w.decoy = decoy
		return index.Response{
			SchemaVersion: index.SchemaVersionV2, DocumentID: req.DocumentID, ProcedureVersionID: req.ProcedureVersionID,
			XAID: req.XAID, JobID: req.JobID, GenerationID: decoy, Outcome: index.OutcomeReady,
			ChunkCount: 1, VectorCount: 1, ManifestHash: "not-the-staged-generation",
		}, nil
	}
	if w.mode == "late" {
		_, _ = w.pool.Exec(ctx, `UPDATE document_index_jobs SET claim_token = $2 WHERE id = $1`, req.JobID, uuid.New())
	}
	text := "xin chao"
	sum := sha256.Sum256([]byte(text))
	textSHA := hex.EncodeToString(sum[:])
	line := "0|1|1|" + textSHA + "|2|native"
	manifestSum := sha256.Sum256([]byte(line))
	manifest := hex.EncodeToString(manifestSum[:])
	source := req.Checksum
	if w.mode == "fail" {
		manifest = hex.EncodeToString(sha256.New().Sum(nil))
	}
	_, err := w.pool.Exec(ctx, `
		UPDATE document_index_generations
		SET manifest_hash = $2, chunk_count = 1, vector_count = 1, page_count = 1,
			native_page_count = 1, ocr_page_count = 0, content_sha256 = $3,
			pipeline_version = 'p4b.1', source_sha256 = $4
		WHERE id = $1 AND status = 'STAGING'`, req.GenerationID, manifest, textSHA, source)
	if err != nil {
		return index.Response{}, err
	}
	chunkID := uuid.New()
	_, err = w.pool.Exec(ctx, `
		INSERT INTO knowledge_chunks (
			id, xa_id, procedure_id, procedure_version_id, document_id, chunk_index, content,
			generation_id, page_start, page_end, text_sha256, token_count, extraction_source
		) VALUES ($1,$2,$3,$4,$5,0,$6,$7,1,1,$8,2,'native')`,
		chunkID, req.XAID, w.procedureID, req.ProcedureVersionID, req.DocumentID, text, req.GenerationID, textSHA)
	if err != nil {
		return index.Response{}, err
	}
	code := ""
	outcome := index.OutcomeReady
	if w.mode == "fail" {
		outcome = index.OutcomeFailed
		code = "qdrant_failed"
	}
	var errPtr *string
	if code != "" {
		errPtr = &code
	}
	return index.Response{
		SchemaVersion: index.SchemaVersionV2, DocumentID: req.DocumentID, ProcedureVersionID: req.ProcedureVersionID,
		XAID: req.XAID, JobID: req.JobID, GenerationID: req.GenerationID, Outcome: outcome, ErrorCode: errPtr,
		SourceSHA256: source, ContentSHA256: textSHA, PagesProcessed: 1, NativePages: 1,
		ChunkCount: 1, VectorCount: 1, ManifestHash: manifest, PipelineVersion: index.PipelineVersion,
		VectorDimension: 384,
	}, nil
}

func pipelineEngine(t *testing.T, pool *pgxpool.Pool, worker index.Worker) (http.Handler, string) {
	t.Helper()
	cfg := config.Config{
		Env: "local", XAID: "xa_chu_se", LogLevel: "error", IndexMode: "pipeline",
		CitizenDomainIDs: []string{"ho_tich_chung_thuc"},
		JWTSecret:        "integration-test-jwt-secret-key", JWTExpireHours: 24,
		AdminIngestionEnabled: true, AdminIndexingEnabled: true,
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

func TestWrongGenerationDoesNotSettleAnotherStagingRow(t *testing.T) {
	pool := requireDB(t)
	defer pool.Close()
	versionID := oneVersion(t, pool, "xa_chu_se", "ho_tich_chung_thuc")
	var procedureID string
	if err := pool.QueryRow(context.Background(), `SELECT procedure_id FROM procedure_versions WHERE id = $1`, versionID).Scan(&procedureID); err != nil {
		t.Fatal(err)
	}
	worker := &scriptWorker{pool: pool, procedureID: procedureID, mode: "wrong"}
	eng, admin := pipelineEngine(t, pool, worker)
	docID := uploadDoc(t, eng, admin, "%PDF-1.4\n%wrong-"+uuid.NewString()+"\n")
	t.Cleanup(func() { cleanupDoc(pool, docID) })
	linkDoc(t, eng, admin, docID, versionID)
	res := call(t, eng, http.MethodPost, "/api/v1/admin/documents/"+docID+"/links/"+versionID+"/index", admin, uuid.New(), nil)
	if res.Code != http.StatusOK || !bytesContains(res.Body.String(), `"error_code":"worker_mismatch"`) {
		t.Fatalf("mismatch %d %s", res.Code, res.Body.String())
	}
	var stagedStatus, decoyStatus string
	if err := pool.QueryRow(context.Background(), `
		SELECT status FROM document_index_generations
		WHERE document_id = $1 AND id <> $2`, docID, worker.decoy).Scan(&stagedStatus); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(context.Background(), `
		SELECT status FROM document_index_generations WHERE id = $1`, worker.decoy).Scan(&decoyStatus); err != nil {
		t.Fatal(err)
	}
	if stagedStatus != "FAILED" || decoyStatus != "STAGING" {
		t.Fatalf("staged %s decoy %s", stagedStatus, decoyStatus)
	}
	var active *string
	if err := pool.QueryRow(context.Background(), `
		SELECT active_generation_id::text FROM procedure_version_documents
		WHERE document_id = $1 AND procedure_version_id = $2`, docID, versionID).Scan(&active); err != nil {
		t.Fatal(err)
	}
	if active != nil {
		t.Fatalf("wrong id published %s", *active)
	}
}

func TestUnlinkBeforeIndexAfterFailedAndAfterReadyThenRelink(t *testing.T) {
	pool := requireDB(t)
	defer pool.Close()
	versionID := oneVersion(t, pool, "xa_chu_se", "ho_tich_chung_thuc")
	var procedureID string
	if err := pool.QueryRow(context.Background(), `SELECT procedure_id FROM procedure_versions WHERE id = $1`, versionID).Scan(&procedureID); err != nil {
		t.Fatal(err)
	}
	worker := &scriptWorker{pool: pool, procedureID: procedureID, mode: "fail"}
	eng, admin := pipelineEngine(t, pool, worker)
	docID := uploadDoc(t, eng, admin, "%PDF-1.4\n%unlink-"+uuid.NewString()+"\n")
	t.Cleanup(func() { cleanupDoc(pool, docID) })
	linkDoc(t, eng, admin, docID, versionID)
	unlink(t, eng, admin, docID, versionID)
	assertSoftUnlinked(t, pool, docID, versionID, "")
	blocked := call(t, eng, http.MethodPost, "/api/v1/admin/documents/"+docID+"/links/"+versionID+"/index", admin, uuid.New(), nil)
	if blocked.Code != http.StatusConflict && blocked.Code != http.StatusNotFound && blocked.Code != http.StatusUnprocessableEntity {
		t.Fatalf("index after unlink %d %s", blocked.Code, blocked.Body.String())
	}
	linkDoc(t, eng, admin, docID, versionID)
	var relinkStatus string
	var relinkActive *string
	if err := pool.QueryRow(context.Background(), `
		SELECT index_status, active_generation_id::text FROM procedure_version_documents
		WHERE document_id = $1 AND procedure_version_id = $2 AND unlinked_at IS NULL`, docID, versionID).Scan(&relinkStatus, &relinkActive); err != nil {
		t.Fatal(err)
	}
	if relinkStatus != "UPLOADED" || relinkActive != nil {
		t.Fatalf("relink status %s active %v", relinkStatus, relinkActive)
	}
	failed := call(t, eng, http.MethodPost, "/api/v1/admin/documents/"+docID+"/links/"+versionID+"/index", admin, uuid.New(), nil)
	if failed.Code != http.StatusOK || !bytesContains(failed.Body.String(), `"link_status":"FAILED"`) {
		t.Fatalf("fail %d %s", failed.Code, failed.Body.String())
	}
	unlink(t, eng, admin, docID, versionID)
	assertSoftUnlinked(t, pool, docID, versionID, "FAILED")
	linkDoc(t, eng, admin, docID, versionID)
	worker.mode = "ok"
	readyRes := call(t, eng, http.MethodPost, "/api/v1/admin/documents/"+docID+"/links/"+versionID+"/index", admin, uuid.New(), nil)
	if readyRes.Code != http.StatusOK || !bytesContains(readyRes.Body.String(), `"link_status":"READY"`) {
		t.Fatalf("ready %d %s", readyRes.Code, readyRes.Body.String())
	}
	var before *string
	if err := pool.QueryRow(context.Background(), `
		SELECT active_generation_id::text FROM procedure_version_documents
		WHERE document_id = $1 AND procedure_version_id = $2`, docID, versionID).Scan(&before); err != nil {
		t.Fatal(err)
	}
	unlink(t, eng, admin, docID, versionID)
	assertSoftUnlinked(t, pool, docID, versionID, "SUPERSEDED")
	linkDoc(t, eng, admin, docID, versionID)
	var afterStatus string
	var afterActive *string
	if err := pool.QueryRow(context.Background(), `
		SELECT index_status, active_generation_id::text FROM procedure_version_documents
		WHERE document_id = $1 AND procedure_version_id = $2 AND unlinked_at IS NULL`, docID, versionID).Scan(&afterStatus, &afterActive); err != nil {
		t.Fatal(err)
	}
	if afterStatus != "UPLOADED" || afterActive != nil {
		t.Fatalf("relink restored generation status %s active %v previous %v", afterStatus, afterActive, before)
	}
	other := otherCommuneDoc(t, pool)
	t.Cleanup(func() { cleanupDoc(pool, other) })
	hidden := call(t, eng, http.MethodDelete, "/api/v1/admin/documents/"+other+"/links/"+versionID, admin, uuid.New(), nil)
	if hidden.Code != http.StatusNotFound {
		t.Fatalf("cross commune unlink %d %s", hidden.Code, hidden.Body.String())
	}
}

func TestUnlinkWhileIndexingStaysConflict(t *testing.T) {
	pool := requireDB(t)
	defer pool.Close()
	versionID := oneVersion(t, pool, "xa_chu_se", "ho_tich_chung_thuc")
	var procedureID string
	if err := pool.QueryRow(context.Background(), `SELECT procedure_id FROM procedure_versions WHERE id = $1`, versionID).Scan(&procedureID); err != nil {
		t.Fatal(err)
	}
	entered := make(chan struct{})
	release := make(chan struct{})
	inner := &scriptWorker{pool: pool, procedureID: procedureID, mode: "ok"}
	worker := workerFn(func(ctx context.Context, req index.Request) (index.Response, error) {
		close(entered)
		<-release
		return inner.Index(ctx, req)
	})
	eng, admin := pipelineEngine(t, pool, worker)
	docID := uploadDoc(t, eng, admin, "%PDF-1.4\n%busy-"+uuid.NewString()+"\n")
	t.Cleanup(func() { cleanupDoc(pool, docID) })
	linkDoc(t, eng, admin, docID, versionID)
	var wg sync.WaitGroup
	wg.Add(1)
	var indexed *httptest.ResponseRecorder
	go func() {
		defer wg.Done()
		indexed = call(t, eng, http.MethodPost, "/api/v1/admin/documents/"+docID+"/links/"+versionID+"/index", admin, uuid.New(), nil)
	}()
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("worker did not start")
	}
	conflict := call(t, eng, http.MethodDelete, "/api/v1/admin/documents/"+docID+"/links/"+versionID, admin, uuid.New(), nil)
	close(release)
	wg.Wait()
	if conflict.Code != http.StatusConflict {
		t.Fatalf("unlink during index %d %s", conflict.Code, conflict.Body.String())
	}
	if indexed == nil || indexed.Code != http.StatusOK {
		t.Fatalf("index after release %#v", indexed)
	}
}

func TestCompositeFKRejectsMismatchedChunkTenant(t *testing.T) {
	pool := requireDB(t)
	defer pool.Close()
	versionID := oneVersion(t, pool, "xa_chu_se", "ho_tich_chung_thuc")
	var procedureID string
	if err := pool.QueryRow(context.Background(), `SELECT procedure_id FROM procedure_versions WHERE id = $1`, versionID).Scan(&procedureID); err != nil {
		t.Fatal(err)
	}
	worker := &scriptWorker{pool: pool, procedureID: procedureID, mode: "ok"}
	eng, admin := pipelineEngine(t, pool, worker)
	docID := uploadDoc(t, eng, admin, "%PDF-1.4\n%fk-"+uuid.NewString()+"\n")
	t.Cleanup(func() { cleanupDoc(pool, docID) })
	linkDoc(t, eng, admin, docID, versionID)
	res := call(t, eng, http.MethodPost, "/api/v1/admin/documents/"+docID+"/links/"+versionID+"/index", admin, uuid.New(), nil)
	if res.Code != http.StatusOK {
		t.Fatalf("index %d %s", res.Code, res.Body.String())
	}
	var generationID string
	if err := pool.QueryRow(context.Background(), `
		SELECT id::text FROM document_index_generations WHERE document_id = $1 AND status = 'READY'`, docID).Scan(&generationID); err != nil {
		t.Fatal(err)
	}
	otherDoc := uploadDoc(t, eng, admin, "%PDF-1.4\n%fk-other-"+uuid.NewString()+"\n")
	t.Cleanup(func() { cleanupDoc(pool, otherDoc) })
	_, err := pool.Exec(context.Background(), `
		INSERT INTO knowledge_chunks (
			id, xa_id, procedure_id, procedure_version_id, document_id, chunk_index, content,
			generation_id, page_start, page_end, text_sha256, token_count, extraction_source
		) VALUES ($1,'xa_chu_se',$2,$3,$4,1,'x',$5,1,1,$6,1,'native')`,
		uuid.New(), procedureID, versionID, otherDoc, generationID, hex.EncodeToString(make([]byte, 32)))
	var pgErr *pgconn.PgError
	if !errors.As(err, &pgErr) || pgErr.ConstraintName != "fk_chunks_generation_link" {
		t.Fatalf("composite fk err %#v", err)
	}
}

func TestReaperFailsExpiredStagingAndKeepsActiveGeneration(t *testing.T) {
	pool := requireDB(t)
	defer pool.Close()
	versionID := oneVersion(t, pool, "xa_chu_se", "ho_tich_chung_thuc")
	var procedureID string
	if err := pool.QueryRow(context.Background(), `SELECT procedure_id FROM procedure_versions WHERE id = $1`, versionID).Scan(&procedureID); err != nil {
		t.Fatal(err)
	}
	worker := &scriptWorker{pool: pool, procedureID: procedureID, mode: "ok"}
	eng, admin := pipelineEngine(t, pool, worker)
	docID := uploadDoc(t, eng, admin, "%PDF-1.4\n%reaper-"+uuid.NewString()+"\n")
	t.Cleanup(func() { cleanupDoc(pool, docID) })
	linkDoc(t, eng, admin, docID, versionID)
	if res := call(t, eng, http.MethodPost, "/api/v1/admin/documents/"+docID+"/links/"+versionID+"/index", admin, uuid.New(), nil); res.Code != http.StatusOK {
		t.Fatalf("index %d %s", res.Code, res.Body.String())
	}
	orphanJob := uuid.New()
	orphanGen := uuid.New()
	textSHA := hex.EncodeToString(make([]byte, 32))
	if _, err := pool.Exec(context.Background(), `
		INSERT INTO document_index_jobs (
			id, document_id, procedure_version_id, xa_id, request_id, payload_hash,
			status, claim_token, claimed_at, claim_expires_at
		) VALUES ($1,$2,$3,'xa_chu_se',$4,'orphan','CLAIMED',$5,now(),now() - interval '1 minute')`,
		orphanJob, docID, versionID, uuid.New(), uuid.New()); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(context.Background(), `
		INSERT INTO document_index_generations (
			id, xa_id, document_id, procedure_version_id, job_id, status,
			pipeline_version, extraction_version, ocr_version, chunk_config_hash,
			embedding_model_id, embedding_revision, embedding_checksum, vector_dimension,
			source_sha256, content_sha256, manifest_hash
		) VALUES ($1,'xa_chu_se',$2,$3,$4,'STAGING','p','p','p','p','p','p','',384,$5,'','')`,
		orphanGen, docID, versionID, orphanJob, textSHA); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(context.Background(), `
		UPDATE document_index_generations SET updated_at = now() - interval '2 hours' WHERE id = $1`, orphanGen); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(context.Background(), `
		INSERT INTO knowledge_chunks (
			id, xa_id, procedure_id, procedure_version_id, document_id, chunk_index, content,
			generation_id, page_start, page_end, text_sha256, token_count, extraction_source
		) VALUES ($1,'xa_chu_se',$2,$3,$4,0,'orphan',$5,1,1,$6,1,'native')`,
		uuid.New(), procedureID, versionID, docID, orphanGen, textSHA); err != nil {
		t.Fatal(err)
	}
	youngJob := uuid.New()
	youngGen := uuid.New()
	if _, err := pool.Exec(context.Background(), `
		INSERT INTO document_index_jobs (
			id, document_id, procedure_version_id, xa_id, request_id, payload_hash,
			status, claim_token, finished_at
		) VALUES ($1,$2,$3,'xa_chu_se',$4,'young','FAILED',$5,now())`,
		youngJob, docID, versionID, uuid.New(), uuid.New()); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(context.Background(), `
		INSERT INTO document_index_generations (
			id, xa_id, document_id, procedure_version_id, job_id, status,
			pipeline_version, extraction_version, ocr_version, chunk_config_hash,
			embedding_model_id, embedding_revision, embedding_checksum, vector_dimension,
			source_sha256, content_sha256, manifest_hash, error_code
		) VALUES ($1,'xa_chu_se',$2,$3,$4,'FAILED','p','p','p','p','p','p','',384,$5,'','','qdrant_failed')`,
		youngGen, docID, versionID, youngJob, textSHA); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(context.Background(), `
		INSERT INTO knowledge_chunks (
			id, xa_id, procedure_id, procedure_version_id, document_id, chunk_index, content,
			generation_id, page_start, page_end, text_sha256, token_count, extraction_source
		) VALUES ($1,'xa_chu_se',$2,$3,$4,0,'young',$5,1,1,$6,1,'native')`,
		uuid.New(), procedureID, versionID, docID, youngGen, textSHA); err != nil {
		t.Fatal(err)
	}
	var active string
	if err := pool.QueryRow(context.Background(), `
		SELECT active_generation_id::text FROM procedure_version_documents
		WHERE document_id = $1 AND procedure_version_id = $2`, docID, versionID).Scan(&active); err != nil {
		t.Fatal(err)
	}
	deletes := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		if bytesContains(r.URL.Path, "/points/delete") && bytesContains(string(raw), orphanGen.String()) {
			deletes++
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"status":"ok","result":{"status":"completed","count":0}}`))
	}))
	defer srv.Close()
	reaperEnv := append(os.Environ(), "INDEX_ORPHAN_GRACE_SECONDS=60", "QDRANT_URL="+srv.URL, "QDRANT_COLLECTION=knowledge_chunks", "INDEX_CLEANUP_BATCH_SIZE=8")
	var out []byte
	cleaned := false
	for attempt := 0; attempt < 40 && !cleaned; attempt++ {
		cmd := exec.Command("go", "run", "./cmd/index-reaper")
		cmd.Dir = filepath.Clean(filepath.Join("..", ".."))
		cmd.Env = reaperEnv
		var err error
		out, err = cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("reaper %v %s", err, out)
		}
		if !bytesContains(string(out), "cleanup_completed=") || !bytesContains(string(out), "cleanup_claimed=") {
			t.Fatalf("reaper output %s", out)
		}
		var state string
		if err := pool.QueryRow(context.Background(), `SELECT cleanup_status FROM document_index_generations WHERE id = $1`, orphanGen).Scan(&state); err != nil {
			t.Fatal(err)
		}
		cleaned = state == "COMPLETED"
	}
	if !cleaned {
		t.Fatalf("orphan was not cleaned within the bounded batches: %s", out)
	}
	var orphanStatus string
	var orphanChunks, activeChunks int
	var activeStatus string
	if err := pool.QueryRow(context.Background(), `SELECT status FROM document_index_generations WHERE id = $1`, orphanGen).Scan(&orphanStatus); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(context.Background(), `SELECT count(*) FROM knowledge_chunks WHERE generation_id = $1`, orphanGen).Scan(&orphanChunks); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(context.Background(), `SELECT status FROM document_index_generations WHERE id = $1`, active).Scan(&activeStatus); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(context.Background(), `SELECT count(*) FROM knowledge_chunks WHERE generation_id = $1`, active).Scan(&activeChunks); err != nil {
		t.Fatal(err)
	}
	if orphanStatus != "FAILED" || orphanChunks != 0 || activeStatus != "READY" || activeChunks < 1 {
		t.Fatalf("orphan %s chunks %d active %s chunks %d", orphanStatus, orphanChunks, activeStatus, activeChunks)
	}
	var youngChunks int
	if err := pool.QueryRow(context.Background(), `SELECT count(*) FROM knowledge_chunks WHERE generation_id = $1`, youngGen).Scan(&youngChunks); err != nil {
		t.Fatal(err)
	}
	if youngChunks != 1 {
		t.Fatalf("young failed chunks removed %d", youngChunks)
	}
	var orphanCleanup, youngCleanup string
	if err := pool.QueryRow(context.Background(), `SELECT cleanup_status FROM document_index_generations WHERE id = $1`, orphanGen).Scan(&orphanCleanup); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(context.Background(), `SELECT cleanup_status FROM document_index_generations WHERE id = $1`, youngGen).Scan(&youngCleanup); err != nil {
		t.Fatal(err)
	}
	if orphanCleanup != "COMPLETED" || youngCleanup != "PENDING" {
		t.Fatalf("cleanup orphan=%s young=%s", orphanCleanup, youngCleanup)
	}
	deletes = 0
	again := exec.Command("go", "run", "./cmd/index-reaper")
	again.Dir = filepath.Clean(filepath.Join("..", ".."))
	again.Env = reaperEnv
	if out, err := again.CombinedOutput(); err != nil {
		t.Fatalf("second reaper %v %s", err, out)
	}
	if deletes != 0 {
		t.Fatalf("completed cleanup was deleted again: %d", deletes)
	}
}

func TestReindexUnlinkConflictThenPublish(t *testing.T) {
	pool := requireDB(t)
	defer pool.Close()
	versionID := oneVersion(t, pool, "xa_chu_se", "ho_tich_chung_thuc")
	var procedureID string
	if err := pool.QueryRow(context.Background(), `SELECT procedure_id FROM procedure_versions WHERE id = $1`, versionID).Scan(&procedureID); err != nil {
		t.Fatal(err)
	}
	worker := &scriptWorker{pool: pool, procedureID: procedureID, mode: "ok"}
	eng, admin := pipelineEngine(t, pool, worker)
	docID := uploadDoc(t, eng, admin, "%PDF-1.4\n%race-"+uuid.NewString()+"\n")
	t.Cleanup(func() { cleanupDoc(pool, docID) })
	linkDoc(t, eng, admin, docID, versionID)
	if res := call(t, eng, http.MethodPost, "/api/v1/admin/documents/"+docID+"/links/"+versionID+"/index", admin, uuid.New(), nil); res.Code != http.StatusOK {
		t.Fatalf("index %d %s", res.Code, res.Body.String())
	}
	worker.entered = make(chan struct{})
	worker.release = make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	var reindexed *httptest.ResponseRecorder
	go func() {
		defer wg.Done()
		reindexed = call(t, eng, http.MethodPost, "/api/v1/admin/documents/"+docID+"/links/"+versionID+"/index/reindex", admin, uuid.New(), nil)
	}()
	select {
	case <-worker.entered:
	case <-time.After(5 * time.Second):
		t.Fatal("reindex did not start")
	}
	conflict := call(t, eng, http.MethodDelete, "/api/v1/admin/documents/"+docID+"/links/"+versionID, admin, uuid.New(), nil)
	close(worker.release)
	wg.Wait()
	if conflict.Code != http.StatusConflict {
		t.Fatalf("unlink during reindex %d %s", conflict.Code, conflict.Body.String())
	}
	if reindexed == nil || reindexed.Code != http.StatusOK || !bytesContains(reindexed.Body.String(), `"link_status":"READY"`) {
		t.Fatalf("reindex after release %v", reindexed)
	}
	var ready int
	var active string
	if err := pool.QueryRow(context.Background(), `
		SELECT count(*) FILTER (WHERE status = 'READY'),
		       (SELECT active_generation_id::text FROM procedure_version_documents
		        WHERE document_id = $1 AND procedure_version_id = $2)
		FROM document_index_generations WHERE document_id = $1`, docID, versionID).Scan(&ready, &active); err != nil {
		t.Fatal(err)
	}
	if ready != 1 || active == "" {
		t.Fatalf("ready %d active %s", ready, active)
	}
}

func TestLateFinishAfterUnlinkDoesNotPublish(t *testing.T) {
	pool := requireDB(t)
	defer pool.Close()
	versionID := oneVersion(t, pool, "xa_chu_se", "ho_tich_chung_thuc")
	var procedureID string
	if err := pool.QueryRow(context.Background(), `SELECT procedure_id FROM procedure_versions WHERE id = $1`, versionID).Scan(&procedureID); err != nil {
		t.Fatal(err)
	}
	worker := &scriptWorker{pool: pool, procedureID: procedureID, mode: "ok"}
	eng, admin := pipelineEngine(t, pool, worker)
	docID := uploadDoc(t, eng, admin, "%PDF-1.4\n%late-unlink-"+uuid.NewString()+"\n")
	t.Cleanup(func() { cleanupDoc(pool, docID) })
	linkDoc(t, eng, admin, docID, versionID)
	if res := call(t, eng, http.MethodPost, "/api/v1/admin/documents/"+docID+"/links/"+versionID+"/index", admin, uuid.New(), nil); res.Code != http.StatusOK {
		t.Fatalf("index %d %s", res.Code, res.Body.String())
	}
	worker.entered = make(chan struct{})
	worker.release = make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	var late *httptest.ResponseRecorder
	go func() {
		defer wg.Done()
		late = call(t, eng, http.MethodPost, "/api/v1/admin/documents/"+docID+"/links/"+versionID+"/index/reindex", admin, uuid.New(), nil)
	}()
	select {
	case <-worker.entered:
	case <-time.After(5 * time.Second):
		t.Fatal("reindex did not start")
	}
	if _, err := pool.Exec(context.Background(), `
		UPDATE document_index_jobs SET claim_expires_at = now() - interval '1 second'
		WHERE document_id = $1 AND status = 'CLAIMED'`, docID); err != nil {
		t.Fatal(err)
	}
	unlink(t, eng, admin, docID, versionID)
	close(worker.release)
	wg.Wait()
	if late == nil || late.Code != http.StatusConflict {
		t.Fatalf("late finish %v", late)
	}
	var active *string
	var unlinked bool
	if err := pool.QueryRow(context.Background(), `
		SELECT active_generation_id::text, unlinked_at IS NOT NULL
		FROM procedure_version_documents
		WHERE document_id = $1 AND procedure_version_id = $2`, docID, versionID).Scan(&active, &unlinked); err != nil {
		t.Fatal(err)
	}
	if !unlinked || active != nil {
		t.Fatalf("published after unlink active=%v unlinked=%v", active, unlinked)
	}
}

func TestExpiredClaimDoesNotPublish(t *testing.T) {
	pool := requireDB(t)
	defer pool.Close()
	versionID := oneVersion(t, pool, "xa_chu_se", "ho_tich_chung_thuc")
	var procedureID string
	if err := pool.QueryRow(context.Background(), `SELECT procedure_id FROM procedure_versions WHERE id = $1`, versionID).Scan(&procedureID); err != nil {
		t.Fatal(err)
	}
	worker := &scriptWorker{pool: pool, procedureID: procedureID, mode: "ok"}
	eng, admin := pipelineEngine(t, pool, worker)
	docID := uploadDoc(t, eng, admin, "%PDF-1.4\n%expired-"+uuid.NewString()+"\n")
	t.Cleanup(func() { cleanupDoc(pool, docID) })
	linkDoc(t, eng, admin, docID, versionID)
	if res := call(t, eng, http.MethodPost, "/api/v1/admin/documents/"+docID+"/links/"+versionID+"/index", admin, uuid.New(), nil); res.Code != http.StatusOK {
		t.Fatalf("index %d %s", res.Code, res.Body.String())
	}
	var before string
	if err := pool.QueryRow(context.Background(), `
		SELECT active_generation_id::text FROM procedure_version_documents
		WHERE document_id = $1 AND procedure_version_id = $2`, docID, versionID).Scan(&before); err != nil {
		t.Fatal(err)
	}
	worker.entered = make(chan struct{})
	worker.release = make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	var late *httptest.ResponseRecorder
	go func() {
		defer wg.Done()
		late = call(t, eng, http.MethodPost, "/api/v1/admin/documents/"+docID+"/links/"+versionID+"/index/reindex", admin, uuid.New(), nil)
	}()
	select {
	case <-worker.entered:
	case <-time.After(5 * time.Second):
		t.Fatal("reindex did not start")
	}
	if _, err := pool.Exec(context.Background(), `
		UPDATE document_index_jobs SET claim_expires_at = now() - interval '1 second'
		WHERE document_id = $1 AND status = 'CLAIMED'`, docID); err != nil {
		t.Fatal(err)
	}
	close(worker.release)
	wg.Wait()
	if late == nil || late.Code != http.StatusConflict {
		t.Fatalf("expired finish %v", late)
	}
	var active string
	var stagedReady int
	if err := pool.QueryRow(context.Background(), `
		SELECT active_generation_id::text FROM procedure_version_documents
		WHERE document_id = $1 AND procedure_version_id = $2`, docID, versionID).Scan(&active); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(context.Background(), `
		SELECT count(*) FROM document_index_generations
		WHERE document_id = $1 AND id <> $2 AND status = 'READY'`, docID, before).Scan(&stagedReady); err != nil {
		t.Fatal(err)
	}
	if active != before || stagedReady != 0 {
		t.Fatalf("expired claim published active=%s before=%s extra=%d", active, before, stagedReady)
	}
}

func TestFinishAfterCleanupClaimDoesNotPublish(t *testing.T) {
	pool := requireDB(t)
	defer pool.Close()
	versionID := oneVersion(t, pool, "xa_chu_se", "ho_tich_chung_thuc")
	var procedureID string
	if err := pool.QueryRow(context.Background(), `SELECT procedure_id FROM procedure_versions WHERE id = $1`, versionID).Scan(&procedureID); err != nil {
		t.Fatal(err)
	}
	worker := &scriptWorker{pool: pool, procedureID: procedureID, mode: "ok"}
	eng, admin := pipelineEngine(t, pool, worker)
	docID := uploadDoc(t, eng, admin, "%PDF-1.4\n%cleanup-claim-"+uuid.NewString()+"\n")
	t.Cleanup(func() { cleanupDoc(pool, docID) })
	linkDoc(t, eng, admin, docID, versionID)
	if res := call(t, eng, http.MethodPost, "/api/v1/admin/documents/"+docID+"/links/"+versionID+"/index", admin, uuid.New(), nil); res.Code != http.StatusOK {
		t.Fatalf("index %d %s", res.Code, res.Body.String())
	}
	var before string
	if err := pool.QueryRow(context.Background(), `
		SELECT active_generation_id::text FROM procedure_version_documents
		WHERE document_id = $1 AND procedure_version_id = $2`, docID, versionID).Scan(&before); err != nil {
		t.Fatal(err)
	}
	worker.entered = make(chan struct{})
	worker.release = make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	var late *httptest.ResponseRecorder
	go func() {
		defer wg.Done()
		late = call(t, eng, http.MethodPost, "/api/v1/admin/documents/"+docID+"/links/"+versionID+"/index/reindex", admin, uuid.New(), nil)
	}()
	select {
	case <-worker.entered:
	case <-time.After(5 * time.Second):
		t.Fatal("reindex did not start")
	}
	if _, err := pool.Exec(context.Background(), `
		UPDATE document_index_generations
		SET cleanup_status = 'CLAIMED', cleanup_claim_token = $2, cleanup_claim_expires_at = now() + interval '5 minutes'
		WHERE document_id = $1 AND status = 'STAGING'`, docID, uuid.New()); err != nil {
		t.Fatal(err)
	}
	close(worker.release)
	wg.Wait()
	if late == nil || late.Code != http.StatusConflict {
		t.Fatalf("finish during cleanup %v", late)
	}
	var active string
	if err := pool.QueryRow(context.Background(), `
		SELECT active_generation_id::text FROM procedure_version_documents
		WHERE document_id = $1 AND procedure_version_id = $2`, docID, versionID).Scan(&active); err != nil {
		t.Fatal(err)
	}
	if active != before {
		t.Fatalf("cleanup claim published active=%s before=%s", active, before)
	}
}

func TestReaperSkipsGenerationPublishedBeforeClaim(t *testing.T) {
	pool := requireDB(t)
	defer pool.Close()
	versionID := oneVersion(t, pool, "xa_chu_se", "ho_tich_chung_thuc")
	var procedureID string
	if err := pool.QueryRow(context.Background(), `SELECT procedure_id FROM procedure_versions WHERE id = $1`, versionID).Scan(&procedureID); err != nil {
		t.Fatal(err)
	}
	worker := &scriptWorker{pool: pool, procedureID: procedureID, mode: "ok"}
	eng, admin := pipelineEngine(t, pool, worker)
	docID := uploadDoc(t, eng, admin, "%PDF-1.4\n%published-"+uuid.NewString()+"\n")
	t.Cleanup(func() { cleanupDoc(pool, docID) })
	linkDoc(t, eng, admin, docID, versionID)
	if res := call(t, eng, http.MethodPost, "/api/v1/admin/documents/"+docID+"/links/"+versionID+"/index", admin, uuid.New(), nil); res.Code != http.StatusOK {
		t.Fatalf("index %d %s", res.Code, res.Body.String())
	}
	var active string
	if err := pool.QueryRow(context.Background(), `
		SELECT active_generation_id::text FROM procedure_version_documents
		WHERE document_id = $1 AND procedure_version_id = $2`, docID, versionID).Scan(&active); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(context.Background(), `
		UPDATE document_index_generations SET updated_at = now() - interval '2 hours' WHERE id = $1`, active); err != nil {
		t.Fatal(err)
	}
	deletes := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		if bytesContains(r.URL.Path, "/points/delete") && bytesContains(string(raw), active) {
			deletes++
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"status":"ok","result":{"status":"completed","count":0}}`))
	}))
	defer srv.Close()
	cmd := exec.Command("go", "run", "./cmd/index-reaper")
	cmd.Dir = filepath.Clean(filepath.Join("..", ".."))
	cmd.Env = append(os.Environ(), "INDEX_ORPHAN_GRACE_SECONDS=60", "QDRANT_URL="+srv.URL, "QDRANT_COLLECTION=knowledge_chunks")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("reaper %v %s", err, out)
	}
	if deletes != 0 {
		t.Fatalf("qdrant deletes %d for active generation", deletes)
	}
	var chunks int
	if err := pool.QueryRow(context.Background(), `SELECT count(*) FROM knowledge_chunks WHERE generation_id = $1`, active).Scan(&chunks); err != nil {
		t.Fatal(err)
	}
	if chunks < 1 {
		t.Fatal("active generation chunks were deleted")
	}
}

func TestTwoReapersClaimOneGeneration(t *testing.T) {
	pool := requireDB(t)
	defer pool.Close()
	versionID := oneVersion(t, pool, "xa_chu_se", "ho_tich_chung_thuc")
	var procedureID string
	if err := pool.QueryRow(context.Background(), `SELECT procedure_id FROM procedure_versions WHERE id = $1`, versionID).Scan(&procedureID); err != nil {
		t.Fatal(err)
	}
	worker := &scriptWorker{pool: pool, procedureID: procedureID, mode: "ok"}
	eng, admin := pipelineEngine(t, pool, worker)
	docID := uploadDoc(t, eng, admin, "%PDF-1.4\n%two-reaper-"+uuid.NewString()+"\n")
	t.Cleanup(func() { cleanupDoc(pool, docID) })
	linkDoc(t, eng, admin, docID, versionID)
	if res := call(t, eng, http.MethodPost, "/api/v1/admin/documents/"+docID+"/links/"+versionID+"/index", admin, uuid.New(), nil); res.Code != http.StatusOK {
		t.Fatalf("index %d %s", res.Code, res.Body.String())
	}
	orphanJob := uuid.New()
	orphanGen := uuid.New()
	textSHA := hex.EncodeToString(make([]byte, 32))
	if _, err := pool.Exec(context.Background(), `
		INSERT INTO document_index_jobs (
			id, document_id, procedure_version_id, xa_id, request_id, payload_hash,
			status, claim_token, claimed_at, claim_expires_at
		) VALUES ($1,$2,$3,'xa_chu_se',$4,'two','CLAIMED',$5,now(),now() - interval '1 minute')`,
		orphanJob, docID, versionID, uuid.New(), uuid.New()); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(context.Background(), `
		INSERT INTO document_index_generations (
			id, xa_id, document_id, procedure_version_id, job_id, status,
			pipeline_version, extraction_version, ocr_version, chunk_config_hash,
			embedding_model_id, embedding_revision, embedding_checksum, vector_dimension,
			source_sha256, content_sha256, manifest_hash
		) VALUES ($1,'xa_chu_se',$2,$3,$4,'STAGING','p','p','p','p','p','p','',384,$5,'','')`,
		orphanGen, docID, versionID, orphanJob, textSHA); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(context.Background(), `
		UPDATE document_index_generations SET updated_at = now() - interval '2 hours' WHERE id = $1`, orphanGen); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(context.Background(), `
		INSERT INTO knowledge_chunks (
			id, xa_id, procedure_id, procedure_version_id, document_id, chunk_index, content,
			generation_id, page_start, page_end, text_sha256, token_count, extraction_source
		) VALUES ($1,'xa_chu_se',$2,$3,$4,0,'orphan',$5,1,1,$6,1,'native')`,
		uuid.New(), procedureID, versionID, docID, orphanGen, textSHA); err != nil {
		t.Fatal(err)
	}
	deletes := 0
	var mu sync.Mutex
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if bytesContains(r.URL.Path, "/points/delete") {
			mu.Lock()
			deletes++
			mu.Unlock()
		}
		_, _ = w.Write([]byte(`{"status":"ok","result":{"status":"completed","count":0}}`))
	}))
	defer srv.Close()
	var wg sync.WaitGroup
	errCh := make(chan error, 2)
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			cmd := exec.Command("go", "run", "./cmd/index-reaper")
			cmd.Dir = filepath.Clean(filepath.Join("..", ".."))
			cmd.Env = append(os.Environ(), "INDEX_ORPHAN_GRACE_SECONDS=60", "QDRANT_URL="+srv.URL, "QDRANT_COLLECTION=knowledge_chunks")
			if out, err := cmd.CombinedOutput(); err != nil {
				errCh <- fmt.Errorf("%v %s", err, out)
			}
		}()
	}
	wg.Wait()
	close(errCh)
	for err := range errCh {
		t.Fatal(err)
	}
	if deletes != 1 {
		t.Fatalf("qdrant deletes %d", deletes)
	}
	var chunks int
	var cleanup string
	if err := pool.QueryRow(context.Background(), `
		SELECT count(*) FROM knowledge_chunks WHERE generation_id = $1`, orphanGen).Scan(&chunks); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(context.Background(), `SELECT cleanup_status FROM document_index_generations WHERE id = $1`, orphanGen).Scan(&cleanup); err != nil {
		t.Fatal(err)
	}
	if chunks != 0 || cleanup != "COMPLETED" {
		t.Fatalf("chunks %d cleanup %s", chunks, cleanup)
	}
}

func assertSoftUnlinked(t *testing.T, pool *pgxpool.Pool, docID, versionID, generationStatus string) {
	t.Helper()
	var active *string
	var unlinked bool
	if err := pool.QueryRow(context.Background(), `
		SELECT active_generation_id::text, unlinked_at IS NOT NULL
		FROM procedure_version_documents
		WHERE document_id = $1 AND procedure_version_id = $2`, docID, versionID).Scan(&active, &unlinked); err != nil {
		t.Fatal(err)
	}
	if !unlinked || active != nil {
		t.Fatalf("unlink pointer active=%v unlinked=%v", active, unlinked)
	}
	if generationStatus == "" {
		return
	}
	var n int
	if err := pool.QueryRow(context.Background(), `
		SELECT count(*) FROM document_index_generations
		WHERE document_id = $1 AND procedure_version_id = $2 AND status = $3`, docID, versionID, generationStatus).Scan(&n); err != nil {
		t.Fatal(err)
	}
	if n < 1 {
		t.Fatalf("expected generation status %s", generationStatus)
	}
}
