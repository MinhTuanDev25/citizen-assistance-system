package drafting

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/index"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/repository"
)

type countingIndexWorker struct {
	pool  *pgxpool.Pool
	calls atomic.Int32
}

func (w *countingIndexWorker) Index(ctx context.Context, req index.Request) (index.Response, error) {
	w.calls.Add(1)
	text := "xin chao"
	sum := sha256.Sum256([]byte(text))
	textSHA := hex.EncodeToString(sum[:])
	pageStart, pageEnd := 10, 10
	line := fmt.Sprintf("0|%d|%d|%s|2|native", pageStart, pageEnd, textSHA)
	manifestSum := sha256.Sum256([]byte(line))
	manifest := hex.EncodeToString(manifestSum[:])
	if _, err := w.pool.Exec(ctx, `
		UPDATE document_index_generations
		SET manifest_hash = $2, chunk_count = 1, vector_count = 1, page_count = 1,
			native_page_count = 1, ocr_page_count = 0, content_sha256 = $3,
			pipeline_version = 'p4b.1', source_sha256 = $4
		WHERE id = $1 AND status = 'STAGING'`, req.GenerationID, manifest, textSHA, req.Checksum); err != nil {
		return index.Response{}, err
	}
	if _, err := w.pool.Exec(ctx, `
		INSERT INTO knowledge_chunks (
			id, xa_id, procedure_id, procedure_version_id, document_id, chunk_index, content,
			generation_id, page_start, page_end, text_sha256, token_count, extraction_source
		) VALUES ($1,$2,$3,$4,$5,0,$6,$7,$8,$9,$10,2,'native')`,
		uuid.New(), req.XAID, req.ProcedureID, req.ProcedureVersionID, req.DocumentID, text, req.GenerationID, pageStart, pageEnd, textSHA); err != nil {
		return index.Response{}, err
	}
	return index.Response{
		SchemaVersion: index.SchemaVersionV2, DocumentID: req.DocumentID, ProcedureVersionID: req.ProcedureVersionID,
		Outcome: index.OutcomeReady, XAID: req.XAID, JobID: req.JobID, GenerationID: req.GenerationID,
		ChunkCount: 1, VectorCount: 1, ManifestHash: manifest, PipelineVersion: index.PipelineVersion,
		SourceSHA256: req.Checksum, ContentSHA256: textSHA, VectorDimension: 384,
	}, nil
}

type stubIndex struct {
	result index.Result
	err    error
}

func (s stubIndex) Link(context.Context, index.LinkInput) (index.Link, bool, error) {
	return index.Link{}, false, nil
}

func (s stubIndex) Request(context.Context, uuid.UUID, uuid.UUID, uuid.UUID, uuid.UUID, bool) (index.Result, error) {
	return s.result, s.err
}

func p4bIndexService(pool *repository.IndexRepo, worker index.Worker) *index.Service {
	return &index.Service{
		XAID: "xa_chu_se", Store: pool, Worker: worker,
		Timeout: 8 * time.Second, ClaimTTL: 30 * time.Second,
		Pipeline: true, Bucket: "cas-p4b-integration-test",
		PublishTimeout: 8 * time.Second,
	}
}

func TestP4BConcurrentActivateWithIndexOneJobOneGeneration(t *testing.T) {
	pool := p4bPool(t)
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	fx := seedApprovedDraft(t, ctx, pool)
	worker := &countingIndexWorker{pool: pool}
	svc := fx.service()
	svc.Index = p4bIndexService(&repository.IndexRepo{Pool: pool}, worker)

	var wg sync.WaitGroup
	type outcome struct {
		res ActivateDraftResult
		err error
	}
	out := make([]outcome, 2)
	wg.Add(2)
	for i := 0; i < 2; i++ {
		i := i
		go func() {
			defer wg.Done()
			res, err := svc.Activate(ctx, fx.admin, uuid.New(), fx.draftID, ActivateDraftRequest{Confirm: true, ExpectedUpdatedAt: &fx.updatedAt})
			out[i] = outcome{res, err}
		}()
	}
	wg.Wait()

	var versionIDs []uuid.UUID
	for i, item := range out {
		if item.err != nil {
			t.Fatalf("caller %d: %v", i, item.err)
		}
		if item.res.Version == nil {
			t.Fatalf("caller %d missing version %+v", i, item.res)
		}
		if item.res.Activated && item.res.BlockedReason == "INDEX_METADATA_PERSIST_FAILED" {
			t.Fatalf("caller %d contradictory Activated+persist failure %+v", i, item.res)
		}
		versionIDs = append(versionIDs, item.res.Version.ID)
	}
	if versionIDs[0] != versionIDs[1] {
		t.Fatalf("duplicate versions %s vs %s", versionIDs[0], versionIDs[1])
	}
	var versions, jobs, gens int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM procedure_versions WHERE source_draft_id=$1`, fx.draftID).Scan(&versions); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM document_index_jobs WHERE procedure_version_id=$1`, versionIDs[0]).Scan(&jobs); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM document_index_generations WHERE procedure_version_id=$1 AND status='READY'`, versionIDs[0]).Scan(&gens); err != nil {
		t.Fatal(err)
	}
	if versions != 1 || jobs != 1 || gens != 1 {
		t.Fatalf("want 1 version/job/READY gen, got versions=%d jobs=%d gens=%d worker=%d", versions, jobs, gens, worker.calls.Load())
	}
	if worker.calls.Load() != 1 {
		t.Fatalf("worker calls=%d want 1", worker.calls.Load())
	}
}

func TestP4BIndexReadyMetadataPersistFailureDoesNotActivate(t *testing.T) {
	pool := p4bPool(t)
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	fx := seedApprovedDraft(t, ctx, pool)
	drafts := &repository.DraftRepo{Pool: pool}
	svc := fx.service()
	svc.Index = p4bIndexService(&repository.IndexRepo{Pool: pool}, &countingIndexWorker{pool: pool})
	svc.Drafts = &failUpdateStatus{DraftRepo: drafts, on: repository.DraftStatusApproved, after: 2, err: errString("injected persist failure")}

	first, err := svc.Activate(ctx, fx.admin, uuid.New(), fx.draftID, ActivateDraftRequest{Confirm: true, ExpectedUpdatedAt: &fx.updatedAt})
	if err != nil {
		t.Fatal(err)
	}
	if first.Activated || first.BlockedReason != "INDEX_METADATA_PERSIST_FAILED" {
		t.Fatalf("want persist failure without activate, got %+v", first)
	}
	if first.Phase != PhaseSourceReady && first.Phase != PhaseIndexingRunning {
		t.Fatalf("phase=%s", first.Phase)
	}
	if first.Draft.IndexRequestID != nil || first.Draft.IndexJobID != nil {
		t.Fatalf("persist failed but Draft leaked unsaved index ids: request=%v job=%v", first.Draft.IndexRequestID, first.Draft.IndexJobID)
	}
	var draftStatus, verStatus string
	if err := pool.QueryRow(ctx, `SELECT status FROM procedure_drafts WHERE id=$1`, fx.draftID).Scan(&draftStatus); err != nil {
		t.Fatal(err)
	}
	if first.Version == nil {
		t.Fatal("version must exist for retry")
	}
	if err := pool.QueryRow(ctx, `SELECT status FROM procedure_versions WHERE id=$1`, first.Version.ID).Scan(&verStatus); err != nil {
		t.Fatal(err)
	}
	if draftStatus != repository.DraftStatusApproved || verStatus == "ACTIVE" {
		t.Fatalf("draft=%s version=%s", draftStatus, verStatus)
	}

	svc.Drafts = drafts
	row, err := drafts.Get(ctx, "xa_chu_se", fx.draftID)
	if err != nil {
		t.Fatal(err)
	}
	second, err := svc.Activate(ctx, fx.admin, uuid.New(), fx.draftID, ActivateDraftRequest{Confirm: true, ExpectedUpdatedAt: &row.UpdatedAt})
	if err != nil {
		t.Fatal(err)
	}
	if second.BlockedReason == "INDEX_METADATA_PERSIST_FAILED" {
		t.Fatalf("retry must not keep persist failure: %+v", second)
	}
	if second.Activated && second.BlockedReason != "" && second.BlockedReason != "DRAFT_PUBLISH_PENDING" {
		t.Fatalf("contradictory retry response %+v", second)
	}
	if second.Version == nil || second.Version.ID != first.Version.ID {
		t.Fatalf("retry created another version %+v", second.Version)
	}
}

func TestP4BIndexPendingMetadataPersistFailure(t *testing.T) {
	pool := p4bPool(t)
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	fx := seedApprovedDraft(t, ctx, pool)
	svc := fx.service()
	svc.Index = stubIndex{result: index.Result{LinkStatus: "PROCESSING", JobStatus: "CLAIMED", JobID: uuid.New()}}
	svc.Drafts = &failUpdateStatus{DraftRepo: &repository.DraftRepo{Pool: pool}, on: repository.DraftStatusApproved, after: 2, err: errString("injected persist failure")}
	out, err := svc.Activate(ctx, fx.admin, uuid.New(), fx.draftID, ActivateDraftRequest{Confirm: true, ExpectedUpdatedAt: &fx.updatedAt})
	if err != nil {
		t.Fatal(err)
	}
	if out.Activated || !out.IndexingPending || out.BlockedReason != "INDEX_METADATA_PERSIST_FAILED" {
		t.Fatalf("got %+v", out)
	}
	if out.Phase != PhaseIndexingRunning {
		t.Fatalf("phase=%s", out.Phase)
	}
	if out.Draft.IndexRequestID != nil || out.Draft.IndexJobID != nil {
		t.Fatalf("persist failed but Draft leaked unsaved index ids: request=%v job=%v", out.Draft.IndexRequestID, out.Draft.IndexJobID)
	}
}

func TestP4BIndexPendingPersistSuccessStaysPending(t *testing.T) {
	pool := p4bPool(t)
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	fx := seedApprovedDraft(t, ctx, pool)
	svc := fx.service()
	svc.Index = stubIndex{result: index.Result{LinkStatus: "PROCESSING", JobStatus: "CLAIMED", JobID: uuid.New()}}
	out, err := svc.Activate(ctx, fx.admin, uuid.New(), fx.draftID, ActivateDraftRequest{Confirm: true, ExpectedUpdatedAt: &fx.updatedAt})
	if err != nil {
		t.Fatal(err)
	}
	if out.Activated || !out.IndexingPending || out.BlockedReason != "INDEXING_PENDING" {
		t.Fatalf("got %+v", out)
	}
}

func TestP4BIndexReadyPersistSuccessDoesNotContradict(t *testing.T) {
	pool := p4bPool(t)
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	fx := seedApprovedDraft(t, ctx, pool)
	svc := fx.service()
	svc.Index = p4bIndexService(&repository.IndexRepo{Pool: pool}, &countingIndexWorker{pool: pool})
	out, err := svc.Activate(ctx, fx.admin, uuid.New(), fx.draftID, ActivateDraftRequest{Confirm: true, ExpectedUpdatedAt: &fx.updatedAt})
	if err != nil {
		t.Fatal(err)
	}
	if out.BlockedReason == "INDEX_METADATA_PERSIST_FAILED" {
		t.Fatalf("persist succeeded but reported failure: %+v", out)
	}
	if out.Activated && out.BlockedReason != "" && out.BlockedReason != "DRAFT_PUBLISH_PENDING" {
		t.Fatalf("contradictory %+v", out)
	}
	if out.Version == nil {
		t.Fatal("missing version")
	}
}
