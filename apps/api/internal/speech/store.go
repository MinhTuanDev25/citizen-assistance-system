package speech

import (
	"context"
	"errors"
	"strings"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
)

const (
	OutcomeSuccess = "SUCCESS"
	OutcomeFailure = "FAILURE"
)

// ErrAuditUnavailable is returned when the store cannot persist (nil pool).
// Handlers must treat this as a best-effort failure, not a citizen 500.
var ErrAuditUnavailable = errors.New("speech: audit store is not configured")

// Record is operational audit metadata. It must not include raw audio or
// full citizen transcripts.
type Record struct {
	RequestID            uuid.UUID
	SessionID            uuid.UUID
	XaID                 string
	UserID               *uuid.UUID
	Outcome              string
	ErrorCode            string
	HTTPStatus           int
	UpstreamStatus       int
	DurationMS           *int
	LatencyMS            *int
	ModelID              string
	TokenizerParity      string
	ResearchProvenance   string
	CheckpointIdentity   string
	GenerationMatchesRQ1 *bool
	ASRCheckpointStep    *int
	MTCheckpointStep     *int
}

type Store struct {
	Pool *pgxpool.Pool
}

func (s *Store) Insert(ctx context.Context, rec Record) error {
	if s == nil || s.Pool == nil {
		return ErrAuditUnavailable
	}
	status := "QUEUED"
	switch rec.Outcome {
	case OutcomeSuccess:
		status = "OK"
	case OutcomeFailure:
		status = "FAILED"
	}
	_, err := s.Pool.Exec(ctx, `
		INSERT INTO speech_translation_requests (
			request_id, session_id, xa_id, user_id, outcome, error_code,
			http_status, upstream_status, duration_ms, latency_ms, model_id,
			tokenizer_parity, research_provenance, checkpoint_identity,
			generation_matches_rq1, asr_checkpoint_step, mt_checkpoint_step,
			status
		) VALUES (
			$1,$2,$3,$4,$5,$6,
			$7,$8,$9,$10,$11,
			$12,$13,$14,
			$15,$16,$17,
			$18
		)`,
		rec.RequestID,
		rec.SessionID,
		nullIfEmpty(rec.XaID),
		rec.UserID,
		nullIfEmpty(rec.Outcome),
		nullIfEmpty(rec.ErrorCode),
		nullIfZero(rec.HTTPStatus),
		nullIfZero(rec.UpstreamStatus),
		rec.DurationMS,
		rec.LatencyMS,
		nullIfEmpty(rec.ModelID),
		nullIfEmpty(rec.TokenizerParity),
		nullIfEmpty(rec.ResearchProvenance),
		nullIfEmpty(rec.CheckpointIdentity),
		rec.GenerationMatchesRQ1,
		rec.ASRCheckpointStep,
		rec.MTCheckpointStep,
		status,
	)
	return err
}

func nullIfEmpty(v string) any {
	if strings.TrimSpace(v) == "" {
		return nil
	}
	return v
}

func nullIfZero(v int) any {
	if v == 0 {
		return nil
	}
	return v
}
