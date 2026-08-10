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

type Session struct {
	ID                        uuid.UUID  `json:"id"`
	UserID                    *uuid.UUID `json:"user_id,omitempty"`
	GuestToken                *string    `json:"guest_token,omitempty"`
	XaID                      string     `json:"xa_id"`
	ActiveProcedureID         *uuid.UUID `json:"active_procedure_id,omitempty"`
	ActiveProcedureVersionID  *uuid.UUID `json:"active_procedure_version_id,omitempty"`
	Status                    string     `json:"status"`
	CreatedAt                 time.Time  `json:"created_at"`
	UpdatedAt                 time.Time  `json:"updated_at"`
}

type Message struct {
	ID        uuid.UUID      `json:"id"`
	SessionID uuid.UUID      `json:"session_id"`
	RequestID uuid.UUID      `json:"request_id"`
	Role      string         `json:"role"`
	Content   string         `json:"content"`
	Action    *string        `json:"action,omitempty"`
	Metadata  map[string]any `json:"metadata,omitempty"`
	CreatedAt time.Time      `json:"created_at"`
}

type SessionRepo struct {
	Pool *pgxpool.Pool
}

func (r *SessionRepo) CreateGuest(ctx context.Context, xaID, guestToken string) (*Session, error) {
	var s Session
	err := r.Pool.QueryRow(ctx, `
		INSERT INTO conversation_sessions (guest_token, xa_id, status)
		VALUES ($1, $2, 'OPEN')
		RETURNING id, user_id, guest_token, xa_id,
		          active_procedure_id, active_procedure_version_id,
		          status, created_at, updated_at`,
		guestToken, xaID,
	).Scan(
		&s.ID, &s.UserID, &s.GuestToken, &s.XaID,
		&s.ActiveProcedureID, &s.ActiveProcedureVersionID,
		&s.Status, &s.CreatedAt, &s.UpdatedAt,
	)
	if err != nil {
		return nil, err
	}
	return &s, nil
}

func (r *SessionRepo) GetByID(ctx context.Context, id uuid.UUID) (*Session, error) {
	var s Session
	err := r.Pool.QueryRow(ctx, `
		SELECT id, user_id, guest_token, xa_id,
		       active_procedure_id, active_procedure_version_id,
		       status, created_at, updated_at
		FROM conversation_sessions
		WHERE id = $1`, id,
	).Scan(
		&s.ID, &s.UserID, &s.GuestToken, &s.XaID,
		&s.ActiveProcedureID, &s.ActiveProcedureVersionID,
		&s.Status, &s.CreatedAt, &s.UpdatedAt,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return &s, nil
}

func (r *SessionRepo) GetByGuestToken(ctx context.Context, guestToken string) (*Session, error) {
	var s Session
	err := r.Pool.QueryRow(ctx, `
		SELECT id, user_id, guest_token, xa_id,
		       active_procedure_id, active_procedure_version_id,
		       status, created_at, updated_at
		FROM conversation_sessions
		WHERE guest_token = $1`, guestToken,
	).Scan(
		&s.ID, &s.UserID, &s.GuestToken, &s.XaID,
		&s.ActiveProcedureID, &s.ActiveProcedureVersionID,
		&s.Status, &s.CreatedAt, &s.UpdatedAt,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return &s, nil
}

func (r *SessionRepo) TouchUpdatedAt(ctx context.Context, id uuid.UUID) error {
	_, err := r.Pool.Exec(ctx, `
		UPDATE conversation_sessions SET updated_at = now() WHERE id = $1`, id)
	return err
}

func (r *SessionRepo) InsertUserMessage(ctx context.Context, sessionID, requestID uuid.UUID, content string) (*Message, error) {
	var m Message
	var meta []byte
	err := r.Pool.QueryRow(ctx, `
		INSERT INTO conversation_messages (session_id, request_id, role, content)
		VALUES ($1, $2, 'USER', $3)
		RETURNING id, session_id, request_id, role, content, action, message_metadata, created_at`,
		sessionID, requestID, content,
	).Scan(&m.ID, &m.SessionID, &m.RequestID, &m.Role, &m.Content, &m.Action, &meta, &m.CreatedAt)
	if err != nil {
		return nil, err
	}
	m.Metadata = decodeMessageMetadata(meta)
	return &m, nil
}

// ListMessages returns messages oldest→newest for chat UI replay.
func (r *SessionRepo) ListMessages(ctx context.Context, sessionID uuid.UUID, limit int) ([]Message, error) {
	if limit <= 0 {
		limit = 100
	}
	if limit > 200 {
		limit = 200
	}
	rows, err := r.Pool.Query(ctx, `
		SELECT id, session_id, request_id, role, content, action, message_metadata, created_at
		FROM conversation_messages
		WHERE session_id = $1
		ORDER BY created_at ASC, id ASC
		LIMIT $2`, sessionID, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := make([]Message, 0)
	for rows.Next() {
		var m Message
		var meta []byte
		if err := rows.Scan(
			&m.ID, &m.SessionID, &m.RequestID, &m.Role, &m.Content, &m.Action, &meta, &m.CreatedAt,
		); err != nil {
			return nil, err
		}
		m.Metadata = decodeMessageMetadata(meta)
		out = append(out, m)
	}
	return out, rows.Err()
}

func decodeMessageMetadata(raw []byte) map[string]any {
	if len(raw) == 0 {
		return map[string]any{}
	}
	var out map[string]any
	if err := json.Unmarshal(raw, &out); err != nil || out == nil {
		return map[string]any{}
	}
	return out
}
