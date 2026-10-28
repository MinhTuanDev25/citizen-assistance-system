//go:build integration

package repository_test

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/activation"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/repository"
)

const seedAdmin = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

func requireP4DB(t *testing.T) *pgxpool.Pool {
	t.Helper()
	url := os.Getenv("CAS_P4_TEST_DATABASE_URL")
	if err := activation.ValidateP4TestDSN(url, os.Getenv); err != nil {
		t.Fatal(err)
	}
	pool, err := pgxpool.New(context.Background(), url)
	if err != nil {
		t.Fatal(err)
	}
	var current string
	if err := pool.QueryRow(context.Background(), `SELECT current_database()`).Scan(&current); err != nil || current != activation.P4TestDatabase {
		pool.Close()
		t.Fatalf("connected to %q, want %s (err=%v)", current, activation.P4TestDatabase, err)
	}
	t.Cleanup(pool.Close)
	return pool
}

type jobFixture struct {
	proc, version, doc, gen uuid.UUID
}

func newJobFixture(t *testing.T, pool *pgxpool.Pool) jobFixture {
	t.Helper()
	ctx := context.Background()
	fx := jobFixture{proc: uuid.New(), version: uuid.New(), doc: uuid.New(), gen: uuid.New()}
	code := "p45_job_" + uuid.NewString()[:8]
	sum := strings.Repeat(strings.ReplaceAll(fx.doc.String(), "-", ""), 2)
	jobID := uuid.New()
	steps := []struct {
		sql  string
		args []any
	}{
		{`INSERT INTO procedures (id, procedure_code, domain_id, name, xa_id) VALUES ($1,$2,'ho_tich_chung_thuc',$2,'xa_chu_se')`, []any{fx.proc, code}},
		{`INSERT INTO procedure_versions (id, procedure_id, version, status, definition, created_by) VALUES ($1,$2,'0.0.0','SOURCE','{}'::jsonb,$3)`, []any{fx.version, fx.proc, seedAdmin}},
		{`INSERT INTO documents (id, xa_id, domain_id, title, filename, storage_uri, checksum, mime_type, file_size_bytes, processing_status, validity_status, uploaded_by)
			VALUES ($1,'xa_chu_se','ho_tich_chung_thuc','P45 job source','p45.pdf','s3://cas-documents/p45',$2,'application/pdf',12,'READY','VALID',$3)`, []any{fx.doc, sum, seedAdmin}},
		{`INSERT INTO procedure_version_documents (procedure_version_id, document_id, procedure_id, xa_id, domain_id, relationship_type, index_status)
			VALUES ($1,$2,$3,'xa_chu_se','ho_tich_chung_thuc','SOURCE','UPLOADED')`, []any{fx.version, fx.doc, fx.proc}},
		{`INSERT INTO document_index_jobs (id, document_id, xa_id, request_id, payload_hash, status, procedure_version_id, claim_token, finished_at)
			VALUES ($1,$2,'xa_chu_se',$3,'hash','SUCCEEDED',$4,gen_random_uuid(),now())`, []any{jobID, fx.doc, uuid.New(), fx.version}},
		{`INSERT INTO document_index_generations (id, xa_id, document_id, procedure_version_id, job_id, status,
			pipeline_version, extraction_version, ocr_version, chunk_config_hash, embedding_model_id, embedding_revision,
			embedding_checksum, vector_dimension, source_sha256, content_sha256, manifest_hash, chunk_count, vector_count, published_at)
			VALUES ($1,'xa_chu_se',$2,$3,$4,'READY','t','t','t','t','e5-small','rev','',384,$5,'','',2,2,now())`, []any{fx.gen, fx.doc, fx.version, jobID, sum}},
	}
	for _, s := range steps {
		if _, err := pool.Exec(ctx, s.sql, s.args...); err != nil {
			t.Fatal(err)
		}
	}
	t.Cleanup(func() {
		ctx := context.Background()
		_, _ = pool.Exec(ctx, `DELETE FROM procedure_draft_jobs WHERE procedure_id = $1`, fx.proc)
		_, _ = pool.Exec(ctx, `DELETE FROM procedure_drafts WHERE procedure_id = $1`, fx.proc)
		_, _ = pool.Exec(ctx, `DELETE FROM document_index_generations WHERE id = $1`, fx.gen)
		_, _ = pool.Exec(ctx, `DELETE FROM procedure_version_documents WHERE procedure_id = $1`, fx.proc)
		_, _ = pool.Exec(ctx, `DELETE FROM document_index_jobs WHERE document_id = $1`, fx.doc)
		_, _ = pool.Exec(ctx, `DELETE FROM documents WHERE id = $1`, fx.doc)
		_, _ = pool.Exec(ctx, `DELETE FROM procedure_versions WHERE procedure_id = $1`, fx.proc)
		_, _ = pool.Exec(ctx, `DELETE FROM procedures WHERE id = $1`, fx.proc)
	})
	return fx
}

func TestDraftJobRepoLifecycle(t *testing.T) {
	pool := requireP4DB(t)
	fx := newJobFixture(t, pool)
	ctx := context.Background()
	repo := &repository.DraftJobRepo{Pool: pool}
	admin := uuid.MustParse(seedAdmin)
	rid := uuid.New()
	ex := [][]uuid.UUID{{uuid.New(), uuid.New()}, {uuid.New()}}
	job := repository.DraftJob{
		XaID: "xa_chu_se", ProcedureID: fx.proc, DocumentID: fx.doc, GenerationID: fx.gen, Mode: "FULL",
		RequestID: rid, RequestHash: "h1", Plan: json.RawMessage(`{"section_title":"t"}`), CreatedBy: admin,
	}
	batches := []repository.DraftJobBatch{
		{BatchIndex: 0, PageStart: 9, PageEnd: 15, ChunkStart: 27, ChunkEnd: 57, ExcerptIDs: ex[0]},
		{BatchIndex: 1, PageStart: 15, PageEnd: 21, ChunkStart: 58, ChunkEnd: 110, ExcerptIDs: ex[1]},
	}
	created, ok, err := repo.Create(ctx, job, batches)
	if err != nil || !ok || created.Status != repository.DraftJobQueued || created.BatchesTotal != 2 {
		t.Fatalf("create: %+v ok=%v err=%v", created, ok, err)
	}
	replay, ok, err := repo.Create(ctx, job, batches)
	if err != nil || ok || replay.ID != created.ID {
		t.Fatalf("replay: id=%s ok=%v err=%v", replay.ID, ok, err)
	}
	changed := job
	changed.RequestHash = "h2"
	if _, _, err := repo.Create(ctx, changed, batches); !errors.Is(err, repository.ErrIdempotency) {
		t.Fatalf("changed body: %v", err)
	}

	ownerA, ownerB := uuid.New(), uuid.New()
	claimed, ok, err := repo.Claim(ctx, "xa_chu_se", ownerA, time.Minute)
	if err != nil || !ok || claimed.ID != created.ID || claimed.Status != repository.DraftJobRunning {
		t.Fatalf("claim: %+v ok=%v err=%v", claimed, ok, err)
	}
	if other, ok, _ := repo.Claim(ctx, "xa_chu_se", ownerB, time.Minute); ok && other.ID == created.ID {
		t.Fatal("a leased job must not be claimed twice")
	}
	if err := repo.StartBatch(ctx, created.ID, ownerB, 0); !errors.Is(err, repository.ErrConflict) {
		t.Fatalf("non-owner start: %v", err)
	}
	if err := repo.StartBatch(ctx, created.ID, ownerA, 0); err != nil {
		t.Fatal(err)
	}
	if err := repo.EndBatch(ctx, created.ID, ownerA, 0, repository.DraftBatchSucceeded, json.RawMessage(`{"definition":{}}`), "", ""); err != nil {
		t.Fatal(err)
	}
	if err := repo.StartBatch(ctx, created.ID, ownerA, 1); err != nil {
		t.Fatal(err)
	}
	if err := repo.EndBatch(ctx, created.ID, ownerA, 1, repository.DraftBatchFailed, nil, "DRAFT_TIMEOUT", "timeout"); err != nil {
		t.Fatal(err)
	}
	got, bs, err := repo.Get(ctx, "xa_chu_se", created.ID)
	if err != nil || got.BatchesDone != 1 || got.BatchesFailed != 1 || len(bs) != 2 || len(bs[0].ExcerptIDs) != 2 || bs[0].ExcerptIDs[1] != ex[0][1] {
		t.Fatalf("counts/batches: %+v %+v err=%v", got, bs, err)
	}
	// SUCCEEDED without a draft is refused by the schema.
	if err := repo.Finish(ctx, created.ID, ownerA, repository.DraftJobSucceeded, nil, "", ""); err == nil {
		t.Fatal("SUCCEEDED without draft_id must violate procedure_draft_jobs_success_check")
	}
	if err := repo.Finish(ctx, created.ID, ownerA, repository.DraftJobFailed, nil, "BATCH_FAILED", "1/2"); err != nil {
		t.Fatal(err)
	}

	retried, err := repo.Retry(ctx, "xa_chu_se", created.ID)
	if err != nil || retried.Status != repository.DraftJobQueued || retried.BatchesDone != 1 || retried.BatchesFailed != 0 {
		t.Fatalf("retry: %+v err=%v", retried, err)
	}
	_, bs, _ = repo.Get(ctx, "xa_chu_se", created.ID)
	if bs[0].Status != repository.DraftBatchSucceeded || len(bs[0].Result) == 0 || bs[1].Status != repository.DraftBatchPending || bs[1].Attempts != 0 {
		t.Fatalf("retry must keep finished results and reset failed batches: %+v", bs)
	}
	if _, err := repo.Retry(ctx, "xa_chu_se", created.ID); !errors.Is(err, repository.ErrConflict) {
		t.Fatalf("a QUEUED job is not retryable: %v", err)
	}

	// Lease expiry: a worker that stopped heart-beating loses the job.
	if _, ok, err := repo.Claim(ctx, "xa_chu_se", ownerA, 500*time.Millisecond); err != nil || !ok {
		t.Fatalf("reclaim: ok=%v err=%v", ok, err)
	}
	time.Sleep(700 * time.Millisecond)
	if ext, _ := repo.ExtendLease(ctx, created.ID, ownerB, time.Minute); ext {
		t.Fatal("a non-owner cannot extend the lease")
	}
	resumed, ok, err := repo.Claim(ctx, "xa_chu_se", ownerB, time.Minute)
	if err != nil || !ok || resumed.ID != created.ID {
		t.Fatalf("expired lease must be claimable: ok=%v err=%v", ok, err)
	}
	if err := repo.StartBatch(ctx, created.ID, ownerB, 1); err != nil {
		t.Fatal(err)
	}
	if err := repo.EndBatch(ctx, created.ID, ownerB, 1, repository.DraftBatchSucceeded, json.RawMessage(`{}`), "", ""); err != nil {
		t.Fatal(err)
	}
	draft := uuid.NewSHA1(created.ID, []byte("aggregate-draft"))
	if _, err := pool.Exec(ctx, `INSERT INTO procedure_drafts (id, document_id, procedure_id, status, created_by, updated_by) VALUES ($1,$2,$3,'DRAFT',$4,$4)`,
		draft, fx.doc, fx.proc, admin); err != nil {
		t.Fatal(err)
	}
	if err := repo.Finish(ctx, created.ID, ownerB, repository.DraftJobSucceeded, &draft, "", ""); err != nil {
		t.Fatal(err)
	}
	final, _, _ := repo.Get(ctx, "xa_chu_se", created.ID)
	if final.Status != repository.DraftJobSucceeded || final.DraftID == nil || *final.DraftID != draft || final.BatchesDone != 2 {
		t.Fatalf("final: %+v", final)
	}
	if list, err := repo.List(ctx, "xa_chu_se", fx.proc, 10); err != nil || len(list) != 1 {
		t.Fatalf("list: %d err=%v", len(list), err)
	}
}
