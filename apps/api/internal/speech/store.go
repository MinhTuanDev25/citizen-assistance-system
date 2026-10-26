package speech

import (
	"context"
	"errors"
	"strings"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
)

const (
	OutcomeSuccess = "SUCCESS"
	OutcomeFailure = "FAILURE"
)

// LinkOutcome is the observable Speech → Chat Turn audit result.
// Handlers must log it; they must not turn FAILED/CONFLICT into HTTP 500
// after a successful Chat Turn.
type LinkOutcome string

const (
	LinkLinked        LinkOutcome = "LINKED"
	LinkAlreadyLinked LinkOutcome = "ALREADY_LINKED"
	LinkNotFound      LinkOutcome = "NOT_FOUND"
	LinkConflict      LinkOutcome = "CONFLICT"
	LinkFailed        LinkOutcome = "FAILED"
)

type LinkResult struct {
	Outcome LinkOutcome
}

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

// LinkChatTurn records the Chat Turn X-Request-ID on a successful speech
// audit row. It matches the explicit speech request_id and the same session.
// Zero rows is not success. Same IDs retry is ALREADY_LINKED. A different
// turn on a linked speech, a FAILURE row, another session, or a turn id
// already used by a different speech is CONFLICT. It never joins by
// timestamp, never overwrites, and never writes transcripts or audio.
func (s *Store) LinkChatTurn(ctx context.Context, speechRequestID, sessionID, chatTurnRequestID uuid.UUID) (LinkResult, error) {
	failed := LinkResult{Outcome: LinkFailed}
	if s == nil || s.Pool == nil {
		return failed, ErrAuditUnavailable
	}
	if speechRequestID == uuid.Nil || sessionID == uuid.Nil || chatTurnRequestID == uuid.Nil {
		return LinkResult{Outcome: LinkNotFound}, nil
	}
	tx, err := s.Pool.Begin(ctx)
	if err != nil {
		return failed, err
	}
	defer func() { _ = tx.Rollback(ctx) }()

	var rowSession *uuid.UUID
	var outcome *string
	var existing *uuid.UUID
	err = tx.QueryRow(ctx, `
		SELECT session_id, outcome, chat_turn_request_id
		FROM speech_translation_requests
		WHERE request_id = $1
		FOR UPDATE`, speechRequestID).Scan(&rowSession, &outcome, &existing)
	if errors.Is(err, pgx.ErrNoRows) {
		return LinkResult{Outcome: LinkNotFound}, nil
	}
	if err != nil {
		return failed, err
	}
	if rowSession == nil || *rowSession != sessionID || outcome == nil || *outcome != OutcomeSuccess {
		return LinkResult{Outcome: LinkConflict}, nil
	}
	if existing != nil {
		if *existing == chatTurnRequestID {
			return LinkResult{Outcome: LinkAlreadyLinked}, nil
		}
		return LinkResult{Outcome: LinkConflict}, nil
	}

	var other uuid.UUID
	err = tx.QueryRow(ctx, `
		SELECT request_id
		FROM speech_translation_requests
		WHERE chat_turn_request_id = $1 AND request_id <> $2
		LIMIT 1`, chatTurnRequestID, speechRequestID).Scan(&other)
	if err == nil {
		return LinkResult{Outcome: LinkConflict}, nil
	}
	if !errors.Is(err, pgx.ErrNoRows) {
		return failed, err
	}

	tag, err := tx.Exec(ctx, `
		UPDATE speech_translation_requests
		SET chat_turn_request_id = $3
		WHERE request_id = $1
		  AND session_id = $2
		  AND outcome = $4
		  AND chat_turn_request_id IS NULL`,
		speechRequestID, sessionID, chatTurnRequestID, OutcomeSuccess)
	if isUniqueViolation(err) {
		return LinkResult{Outcome: LinkConflict}, nil
	}
	if err != nil {
		return failed, err
	}
	if tag.RowsAffected() != 1 {
		return LinkResult{Outcome: LinkConflict}, nil
	}
	if err := tx.Commit(ctx); err != nil {
		return failed, err
	}
	return LinkResult{Outcome: LinkLinked}, nil
}

func isUniqueViolation(err error) bool {
	var pgErr *pgconn.PgError
	return errors.As(err, &pgErr) && pgErr.Code == "23505"
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
