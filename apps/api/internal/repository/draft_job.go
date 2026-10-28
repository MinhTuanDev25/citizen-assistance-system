package repository

import (
	"context"
	"encoding/json"
	"errors"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

const (
	DraftJobQueued    = "QUEUED"
	DraftJobRunning   = "RUNNING"
	DraftJobSucceeded = "SUCCEEDED"
	DraftJobFailed    = "FAILED"

	DraftBatchPending   = "PENDING"
	DraftBatchRunning   = "RUNNING"
	DraftBatchSucceeded = "SUCCEEDED"
	DraftBatchFailed    = "FAILED"
)

// ErrIdempotency: the request id was already used for a different job body.
var ErrIdempotency = errors.New("idempotency_conflict")

type DraftJob struct {
	ID            uuid.UUID       `json:"id"`
	XaID          string          `json:"-"`
	ProcedureID   uuid.UUID       `json:"procedure_id"`
	DocumentID    uuid.UUID       `json:"document_id"`
	GenerationID  uuid.UUID       `json:"generation_id"`
	Mode          string          `json:"mode"`
	BaseDraftID   *uuid.UUID      `json:"base_draft_id,omitempty"`
	RequestID     uuid.UUID       `json:"request_id"`
	RequestHash   string          `json:"-"`
	Status        string          `json:"status"`
	Plan          json.RawMessage `json:"plan"`
	BatchesTotal  int             `json:"batches_total"`
	BatchesDone   int             `json:"batches_done"`
	BatchesFailed int             `json:"batches_failed"`
	DraftID       *uuid.UUID      `json:"draft_id,omitempty"`
	ErrorCode     *string         `json:"error_code,omitempty"`
	ErrorMessage  *string         `json:"error_message,omitempty"`
	CreatedBy     uuid.UUID       `json:"created_by"`
	CreatedAt     time.Time       `json:"created_at"`
	UpdatedAt     time.Time       `json:"updated_at"`
	StartedAt     *time.Time      `json:"started_at,omitempty"`
	FinishedAt    *time.Time      `json:"finished_at,omitempty"`
}

type DraftJobBatch struct {
	JobID        uuid.UUID       `json:"-"`
	BatchIndex   int             `json:"batch_index"`
	PageStart    int             `json:"page_start"`
	PageEnd      int             `json:"page_end"`
	ChunkStart   int             `json:"chunk_start"`
	ChunkEnd     int             `json:"chunk_end"`
	ExcerptIDs   []uuid.UUID     `json:"-"`
	Excerpts     int             `json:"excerpts"`
	Status       string          `json:"status"`
	Attempts     int             `json:"attempts"`
	Result       json.RawMessage `json:"-"`
	ErrorCode    *string         `json:"error_code,omitempty"`
	ErrorMessage *string         `json:"error_message,omitempty"`
	StartedAt    *time.Time      `json:"started_at,omitempty"`
	FinishedAt   *time.Time      `json:"finished_at,omitempty"`
}

type DraftJobRepo struct {
	Pool *pgxpool.Pool
}

const draftJobColumns = `id, xa_id, procedure_id, document_id, generation_id, mode, base_draft_id,
	request_id, request_hash, status, plan, batches_total, batches_done, batches_failed,
	draft_id, error_code, error_message, created_by, created_at, updated_at, started_at, finished_at`

func scanDraftJob(row pgx.Row) (DraftJob, error) {
	var j DraftJob
	err := row.Scan(&j.ID, &j.XaID, &j.ProcedureID, &j.DocumentID, &j.GenerationID, &j.Mode, &j.BaseDraftID,
		&j.RequestID, &j.RequestHash, &j.Status, &j.Plan, &j.BatchesTotal, &j.BatchesDone, &j.BatchesFailed,
		&j.DraftID, &j.ErrorCode, &j.ErrorMessage, &j.CreatedBy, &j.CreatedAt, &j.UpdatedAt, &j.StartedAt, &j.FinishedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return DraftJob{}, ErrNotFound
	}
	return j, err
}

// Create inserts a job with its planned batches. Replaying the same request id
// with the same body returns the existing job (created=false).
func (r *DraftJobRepo) Create(ctx context.Context, job DraftJob, batches []DraftJobBatch) (DraftJob, bool, error) {
	tx, err := r.Pool.Begin(ctx)
	if err != nil {
		return DraftJob{}, false, err
	}
	defer tx.Rollback(ctx)
	saved, err := scanDraftJob(tx.QueryRow(ctx, `
		INSERT INTO procedure_draft_jobs (
			xa_id, procedure_id, document_id, generation_id, mode, base_draft_id,
			request_id, request_hash, status, plan, batches_total, created_by
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 'QUEUED', $9, $10, $11)
		ON CONFLICT (xa_id, request_id) DO NOTHING
		RETURNING `+draftJobColumns,
		job.XaID, job.ProcedureID, job.DocumentID, job.GenerationID, job.Mode, job.BaseDraftID,
		job.RequestID, job.RequestHash, job.Plan, len(batches), job.CreatedBy))
	if errors.Is(err, ErrNotFound) {
		existing, getErr := scanDraftJob(tx.QueryRow(ctx,
			`SELECT `+draftJobColumns+` FROM procedure_draft_jobs WHERE xa_id = $1 AND request_id = $2`,
			job.XaID, job.RequestID))
		if getErr != nil {
			return DraftJob{}, false, getErr
		}
		if existing.RequestHash != job.RequestHash {
			return DraftJob{}, false, ErrIdempotency
		}
		return existing, false, nil
	}
	if err != nil {
		return DraftJob{}, false, err
	}
	for _, b := range batches {
		if _, err := tx.Exec(ctx, `
			INSERT INTO procedure_draft_job_batches
				(job_id, batch_index, page_start, page_end, chunk_start, chunk_end, excerpt_ids)
			VALUES ($1, $2, $3, $4, $5, $6, $7)`,
			saved.ID, b.BatchIndex, b.PageStart, b.PageEnd, b.ChunkStart, b.ChunkEnd, b.ExcerptIDs); err != nil {
			return DraftJob{}, false, err
		}
	}
	if err := tx.Commit(ctx); err != nil {
		return DraftJob{}, false, err
	}
	return saved, true, nil
}

func (r *DraftJobRepo) Get(ctx context.Context, xaID string, jobID uuid.UUID) (DraftJob, []DraftJobBatch, error) {
	job, err := scanDraftJob(r.Pool.QueryRow(ctx,
		`SELECT `+draftJobColumns+` FROM procedure_draft_jobs WHERE id = $1 AND xa_id = $2`, jobID, xaID))
	if err != nil {
		return DraftJob{}, nil, err
	}
	batches, err := r.batches(ctx, jobID)
	return job, batches, err
}

func (r *DraftJobRepo) batches(ctx context.Context, jobID uuid.UUID) ([]DraftJobBatch, error) {
	rows, err := r.Pool.Query(ctx, `
		SELECT job_id, batch_index, page_start, page_end, chunk_start, chunk_end, excerpt_ids,
		       status, attempts, result, error_code, error_message, started_at, finished_at
		FROM procedure_draft_job_batches WHERE job_id = $1 ORDER BY batch_index`, jobID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := make([]DraftJobBatch, 0)
	for rows.Next() {
		var b DraftJobBatch
		if err := rows.Scan(&b.JobID, &b.BatchIndex, &b.PageStart, &b.PageEnd, &b.ChunkStart, &b.ChunkEnd, &b.ExcerptIDs,
			&b.Status, &b.Attempts, &b.Result, &b.ErrorCode, &b.ErrorMessage, &b.StartedAt, &b.FinishedAt); err != nil {
			return nil, err
		}
		b.Excerpts = len(b.ExcerptIDs)
		out = append(out, b)
	}
	return out, rows.Err()
}

func (r *DraftJobRepo) List(ctx context.Context, xaID string, procedureID uuid.UUID, limit int) ([]DraftJob, error) {
	if limit <= 0 || limit > 100 {
		limit = 20
	}
	rows, err := r.Pool.Query(ctx, `SELECT `+draftJobColumns+` FROM procedure_draft_jobs
		WHERE xa_id = $1 AND procedure_id = $2 ORDER BY created_at DESC LIMIT $3`, xaID, procedureID, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := make([]DraftJob, 0)
	for rows.Next() {
		j, err := scanDraftJob(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, j)
	}
	return out, rows.Err()
}

// Claim takes the oldest QUEUED job, or a RUNNING one whose worker lease expired
// (API restart), and leases it to owner.
func (r *DraftJobRepo) Claim(ctx context.Context, xaID string, owner uuid.UUID, lease time.Duration) (DraftJob, bool, error) {
	job, err := scanDraftJob(r.Pool.QueryRow(ctx, `
		UPDATE procedure_draft_jobs SET
			status = 'RUNNING', lease_owner = $2, lease_expires_at = now() + make_interval(secs => $3),
			started_at = COALESCE(started_at, now()), updated_at = now()
		WHERE id = (
			SELECT id FROM procedure_draft_jobs
			WHERE xa_id = $1 AND status IN ('QUEUED', 'RUNNING')
			  AND (lease_expires_at IS NULL OR lease_expires_at < now())
			ORDER BY created_at
			FOR UPDATE SKIP LOCKED
			LIMIT 1)
		RETURNING `+draftJobColumns, xaID, owner, lease.Seconds()))
	if errors.Is(err, ErrNotFound) {
		return DraftJob{}, false, nil
	}
	if err != nil {
		return DraftJob{}, false, err
	}
	return job, true, nil
}

// ExtendLease keeps a claimed job; false means the lease was lost.
func (r *DraftJobRepo) ExtendLease(ctx context.Context, jobID, owner uuid.UUID, lease time.Duration) (bool, error) {
	tag, err := r.Pool.Exec(ctx, `
		UPDATE procedure_draft_jobs SET lease_expires_at = now() + make_interval(secs => $3), updated_at = now()
		WHERE id = $1 AND lease_owner = $2 AND status = 'RUNNING'`, jobID, owner, lease.Seconds())
	if err != nil {
		return false, err
	}
	return tag.RowsAffected() == 1, nil
}

// StartBatch marks a batch RUNNING and counts the attempt.
func (r *DraftJobRepo) StartBatch(ctx context.Context, jobID, owner uuid.UUID, index int) error {
	tag, err := r.Pool.Exec(ctx, `
		UPDATE procedure_draft_job_batches b SET
			status = 'RUNNING', attempts = attempts + 1, started_at = now(),
			error_code = NULL, error_message = NULL
		FROM procedure_draft_jobs j
		WHERE b.job_id = $1 AND b.batch_index = $3 AND b.status IN ('PENDING', 'RUNNING')
		  AND j.id = b.job_id AND j.lease_owner = $2 AND j.status = 'RUNNING'`, jobID, owner, index)
	if err != nil {
		return err
	}
	if tag.RowsAffected() != 1 {
		return ErrConflict
	}
	return nil
}

// EndBatch stores a batch outcome: SUCCEEDED with its result, PENDING (will be
// retried) or FAILED, and refreshes the job counters in the same transaction.
func (r *DraftJobRepo) EndBatch(ctx context.Context, jobID, owner uuid.UUID, index int, status string, result json.RawMessage, code, message string) error {
	tx, err := r.Pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx)
	var errCode, errMsg *string
	if code != "" {
		errCode, errMsg = &code, &message
	}
	tag, err := tx.Exec(ctx, `
		UPDATE procedure_draft_job_batches b SET
			status = $4, result = COALESCE($5, b.result), error_code = $6, error_message = $7,
			finished_at = CASE WHEN $4 IN ('SUCCEEDED', 'FAILED') THEN now() ELSE NULL END
		FROM procedure_draft_jobs j
		WHERE b.job_id = $1 AND b.batch_index = $3 AND b.status = 'RUNNING'
		  AND j.id = b.job_id AND j.lease_owner = $2 AND j.status = 'RUNNING'`,
		jobID, owner, index, status, result, errCode, errMsg)
	if err != nil {
		return err
	}
	if tag.RowsAffected() != 1 {
		return ErrConflict
	}
	if err := refreshCounts(ctx, tx, jobID); err != nil {
		return err
	}
	return tx.Commit(ctx)
}

func refreshCounts(ctx context.Context, tx pgx.Tx, jobID uuid.UUID) error {
	_, err := tx.Exec(ctx, `
		UPDATE procedure_draft_jobs j SET
			batches_done = (SELECT count(*) FROM procedure_draft_job_batches WHERE job_id = j.id AND status = 'SUCCEEDED'),
			batches_failed = (SELECT count(*) FROM procedure_draft_job_batches WHERE job_id = j.id AND status = 'FAILED'),
			updated_at = now()
		WHERE j.id = $1`, jobID)
	return err
}

// Finish ends a leased job. SUCCEEDED requires draftID (enforced by a CHECK).
func (r *DraftJobRepo) Finish(ctx context.Context, jobID, owner uuid.UUID, status string, draftID *uuid.UUID, code, message string) error {
	var errCode, errMsg *string
	if code != "" {
		errCode, errMsg = &code, &message
	}
	tag, err := r.Pool.Exec(ctx, `
		UPDATE procedure_draft_jobs SET
			status = $3, draft_id = COALESCE($4, draft_id), error_code = $5, error_message = $6,
			finished_at = now(), updated_at = now(), lease_owner = NULL, lease_expires_at = NULL
		WHERE id = $1 AND lease_owner = $2 AND status = 'RUNNING'`,
		jobID, owner, status, draftID, errCode, errMsg)
	if err != nil {
		return err
	}
	if tag.RowsAffected() != 1 {
		return ErrConflict
	}
	return nil
}

// Retry re-queues a FAILED job. Only FAILED batches go back to PENDING; finished
// batches keep their stored results.
func (r *DraftJobRepo) Retry(ctx context.Context, xaID string, jobID uuid.UUID) (DraftJob, error) {
	tx, err := r.Pool.Begin(ctx)
	if err != nil {
		return DraftJob{}, err
	}
	defer tx.Rollback(ctx)
	job, err := scanDraftJob(tx.QueryRow(ctx,
		`SELECT `+draftJobColumns+` FROM procedure_draft_jobs WHERE id = $1 AND xa_id = $2 FOR UPDATE`, jobID, xaID))
	if err != nil {
		return DraftJob{}, err
	}
	if job.Status != DraftJobFailed {
		return DraftJob{}, ErrConflict
	}
	if _, err := tx.Exec(ctx, `
		UPDATE procedure_draft_job_batches SET status = 'PENDING', attempts = 0,
			error_code = NULL, error_message = NULL, started_at = NULL, finished_at = NULL
		WHERE job_id = $1 AND status IN ('FAILED', 'RUNNING', 'PENDING')`, jobID); err != nil {
		return DraftJob{}, err
	}
	if _, err := tx.Exec(ctx, `
		UPDATE procedure_draft_jobs SET status = 'QUEUED', error_code = NULL, error_message = NULL,
			finished_at = NULL, lease_owner = NULL, lease_expires_at = NULL, updated_at = now()
		WHERE id = $1`, jobID); err != nil {
		return DraftJob{}, err
	}
	if err := refreshCounts(ctx, tx, jobID); err != nil {
		return DraftJob{}, err
	}
	if err := tx.Commit(ctx); err != nil {
		return DraftJob{}, err
	}
	job, _, err = r.Get(ctx, xaID, jobID)
	return job, err
}
