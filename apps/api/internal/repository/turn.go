package repository

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
)

// ErrSessionNotOpen is returned when a chat turn targets a closed session.
var ErrSessionNotOpen = errors.New("session_not_open")

// ErrDuplicateTurn is returned when (session_id, request_id) USER already exists.
var ErrDuplicateTurn = errors.New("duplicate_turn")

// ErrIdempotencyConflict is same request_id with a different payload hash.
var ErrIdempotencyConflict = errors.New("idempotency_conflict")

// Idempotency claim states. PENDING is an in-flight extractor lease;
// COMPLETE is a stored turn envelope safe to replay.
const (
	IdempotencyPending  = "PENDING"
	IdempotencyComplete = "COMPLETE"
)

// ClaimResult is the outcome of ResolveExtractClaim.
type ClaimResult int

const (
	// ClaimOwned means this caller inserted or stole the PENDING lease and
	// must call the extractor exactly once.
	ClaimOwned ClaimResult = iota
	// ClaimWait means another caller holds an unexpired lease for the same
	// payload. This caller must not call the extractor.
	ClaimWait
	// ClaimConflict means the same request_id is already claimed or completed
	// with a different payload hash.
	ClaimConflict
)

// IdempotencyState is one turn_idempotency row, including in-flight claims.
type IdempotencyState struct {
	Status       string
	PayloadHash  string
	ResponseJSON []byte
	ExpiresAt    *time.Time
}

// ProcedureDefinitionRow is an ACTIVE (or pinned) procedure plus its definition JSON.
type ProcedureDefinitionRow struct {
	ProcedureID   uuid.UUID
	VersionID     uuid.UUID
	ProcedureCode string
	Name          string
	Version       string
	Definition    json.RawMessage
}

// TurnSnapshot is the locked session plus the catalog data a turn needs.
type TurnSnapshot struct {
	Session       Session
	Catalog       []ProcedureDefinitionRow
	Pinned        *ProcedureDefinitionRow
	PriorRaw      json.RawMessage
	HadPrior      bool
	PendingIntent json.RawMessage
}

// SaveTurnInput is what one successful decision writes.
type SaveTurnInput struct {
	RequestID          uuid.UUID
	ActorUserID        *uuid.UUID
	UserContent        string
	AssistantContent   string
	Action             string
	Metadata           map[string]any
	ProcedureID        *uuid.UUID
	VersionID          *uuid.UUID
	SlotState          map[string]any
	Citations          []map[string]any
	PendingIntent      json.RawMessage // nil = leave unchanged; empty object clears
	ClearPendingIntent bool
	AuditPayload       map[string]any
	PinProcedure       bool // false for CONFIRM_INTENT / OUT_OF_SCOPE without pin change
}

// LockTurn locks the session row and loads catalog, pin, and slot state.
func (r *SessionRepo) LockTurn(ctx context.Context, tx pgx.Tx, sessionID uuid.UUID, citizenDomains []string) (*TurnSnapshot, error) {
	var snap TurnSnapshot
	err := tx.QueryRow(ctx, `
		SELECT id, user_id, guest_token, xa_id,
		       active_procedure_id, active_procedure_version_id,
		       status, created_at, updated_at, pending_intent
		FROM conversation_sessions
		WHERE id = $1
		FOR UPDATE`, sessionID,
	).Scan(
		&snap.Session.ID, &snap.Session.UserID, &snap.Session.GuestToken, &snap.Session.XaID,
		&snap.Session.ActiveProcedureID, &snap.Session.ActiveProcedureVersionID,
		&snap.Session.Status, &snap.Session.CreatedAt, &snap.Session.UpdatedAt,
		&snap.PendingIntent,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, ErrNotFound
	}
	if err != nil {
		return nil, err
	}

	if len(citizenDomains) == 0 {
		citizenDomains = []string{"ho_tich_chung_thuc"}
	}

	rows, err := tx.Query(ctx, `
		SELECT p.id, pv.id, p.procedure_code, p.name, pv.version, pv.definition
		FROM procedures p
		JOIN procedure_versions pv
		  ON pv.id = p.active_version_id
		 AND pv.procedure_id = p.id
		WHERE p.xa_id = $1
		  AND p.domain_id = ANY($2)
		  AND p.active_version_id IS NOT NULL
		  AND pv.status = 'ACTIVE'
		ORDER BY p.procedure_code`, snap.Session.XaID, citizenDomains)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	for rows.Next() {
		var row ProcedureDefinitionRow
		if err := rows.Scan(
			&row.ProcedureID, &row.VersionID, &row.ProcedureCode, &row.Name, &row.Version, &row.Definition,
		); err != nil {
			return nil, err
		}
		snap.Catalog = append(snap.Catalog, row)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}

	if snap.Session.ActiveProcedureID != nil && snap.Session.ActiveProcedureVersionID != nil {
		var pinned ProcedureDefinitionRow
		err = tx.QueryRow(ctx, `
			SELECT p.id, pv.id, p.procedure_code, p.name, pv.version, pv.definition
			FROM procedure_versions pv
			JOIN procedures p ON p.id = pv.procedure_id
			WHERE pv.procedure_id = $1
			  AND pv.id = $2`,
			*snap.Session.ActiveProcedureID, *snap.Session.ActiveProcedureVersionID,
		).Scan(
			&pinned.ProcedureID, &pinned.VersionID, &pinned.ProcedureCode, &pinned.Name, &pinned.Version, &pinned.Definition,
		)
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, ErrNotFound
		}
		if err != nil {
			return nil, err
		}
		snap.Pinned = &pinned

		err = tx.QueryRow(ctx, `
			SELECT slot_state
			FROM session_slot_states
			WHERE session_id = $1 AND procedure_id = $2`,
			snap.Session.ID, pinned.ProcedureID,
		).Scan(&snap.PriorRaw)
		if errors.Is(err, pgx.ErrNoRows) {
			snap.HadPrior = false
		} else if err != nil {
			return nil, err
		} else {
			snap.HadPrior = true
		}
	}
	return &snap, nil
}

// FindCompletedTurn returns USER+ASSISTANT for an idempotent retry.
func (r *SessionRepo) FindCompletedTurn(ctx context.Context, tx pgx.Tx, sessionID, requestID uuid.UUID) (*Message, *Message, error) {
	var user, asst Message
	var userMeta, asstMeta []byte
	err := tx.QueryRow(ctx, `
		SELECT id, session_id, request_id, role, content, action, message_metadata, created_at
		FROM conversation_messages
		WHERE session_id = $1 AND request_id = $2 AND role = 'USER'
		LIMIT 1`, sessionID, requestID,
	).Scan(&user.ID, &user.SessionID, &user.RequestID, &user.Role, &user.Content, &user.Action, &userMeta, &user.CreatedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, nil, nil
	}
	if err != nil {
		return nil, nil, err
	}
	user.Metadata = decodeMessageMetadata(userMeta)

	err = tx.QueryRow(ctx, `
		SELECT id, session_id, request_id, role, content, action, message_metadata, created_at
		FROM conversation_messages
		WHERE session_id = $1 AND request_id = $2 AND role = 'ASSISTANT'
		ORDER BY created_at ASC, id ASC
		LIMIT 1`, sessionID, requestID,
	).Scan(&asst.ID, &asst.SessionID, &asst.RequestID, &asst.Role, &asst.Content, &asst.Action, &asstMeta, &asst.CreatedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return &user, nil, nil
	}
	if err != nil {
		return nil, nil, err
	}
	asst.Metadata = decodeMessageMetadata(asstMeta)
	return &user, &asst, nil
}

// SaveTurn writes the user message, assistant message, slot state, pin, citations, and audit.
func (r *SessionRepo) SaveTurn(ctx context.Context, tx pgx.Tx, sessionID uuid.UUID, in SaveTurnInput) (*Message, *Message, error) {
	userMsg, err := insertTurnMessage(ctx, tx, sessionID, in.RequestID, "USER", in.UserContent, nil, map[string]any{})
	if err != nil {
		if isUniqueViolation(err) {
			return nil, nil, ErrDuplicateTurn
		}
		return nil, nil, err
	}
	asstMsg, err := insertTurnMessage(ctx, tx, sessionID, in.RequestID, "ASSISTANT", in.AssistantContent, &in.Action, in.Metadata)
	if err != nil {
		return nil, nil, err
	}

	for _, cite := range in.Citations {
		if err := insertCitation(ctx, tx, asstMsg.ID, cite); err != nil {
			return nil, nil, err
		}
	}

	if in.PinProcedure && in.ProcedureID != nil && in.VersionID != nil {
		raw, err := json.Marshal(in.SlotState)
		if err != nil {
			return nil, nil, err
		}
		if _, err := tx.Exec(ctx, `
			INSERT INTO session_slot_states (session_id, procedure_id, procedure_version_id, slot_state)
			VALUES ($1, $2, $3, $4::jsonb)
			ON CONFLICT (session_id, procedure_id) DO UPDATE
			SET procedure_version_id = EXCLUDED.procedure_version_id,
			    slot_state = EXCLUDED.slot_state,
			    updated_at = now()`,
			sessionID, *in.ProcedureID, *in.VersionID, raw,
		); err != nil {
			return nil, nil, err
		}
		if _, err := tx.Exec(ctx, `
			UPDATE conversation_sessions
			SET active_procedure_id = $2,
			    active_procedure_version_id = $3,
			    updated_at = now()
			WHERE id = $1`,
			sessionID, *in.ProcedureID, *in.VersionID,
		); err != nil {
			return nil, nil, err
		}
	} else if _, err := tx.Exec(ctx, `
		UPDATE conversation_sessions SET updated_at = now() WHERE id = $1`, sessionID,
	); err != nil {
		return nil, nil, err
	}

	if in.ClearPendingIntent {
		if _, err := tx.Exec(ctx, `
			UPDATE conversation_sessions SET pending_intent = '{}'::jsonb WHERE id = $1`, sessionID,
		); err != nil {
			return nil, nil, err
		}
	} else if len(in.PendingIntent) > 0 {
		if _, err := tx.Exec(ctx, `
			UPDATE conversation_sessions SET pending_intent = $2::jsonb WHERE id = $1`,
			sessionID, in.PendingIntent,
		); err != nil {
			return nil, nil, err
		}
	}

	audit, err := json.Marshal(in.AuditPayload)
	if err != nil {
		return nil, nil, err
	}
	if _, err := tx.Exec(ctx, `
		INSERT INTO audit_logs (request_id, actor_user_id, action, entity_type, entity_id, payload)
		VALUES ($1, $2, 'chat_decision', 'conversation_session', $3, $4::jsonb)`,
		in.RequestID, in.ActorUserID, sessionID.String(), audit,
	); err != nil {
		return nil, nil, err
	}
	return userMsg, asstMsg, nil
}

func insertCitation(ctx context.Context, tx pgx.Tx, messageID uuid.UUID, cite map[string]any) error {
	title, _ := cite["title"].(string)
	sourceType, _ := cite["source_type"].(string)
	docID, _ := cite["doc_id"].(string)
	eff, _ := cite["effective_date"].(string)
	issuer, _ := cite["issuer"].(string)
	page, _ := cite["page_range"].(string)
	_, err := tx.Exec(ctx, `
		INSERT INTO message_citations (message_id, title, source_type, doc_id, effective_date, issuer, page_range)
		VALUES ($1, $2, $3, $4, $5, $6, $7)`,
		messageID, nullIfEmpty(title), nullIfEmpty(sourceType), nullIfEmpty(docID), nullIfEmpty(eff), nullIfEmpty(issuer), nullIfEmpty(page),
	)
	return err
}

func nullIfEmpty(s string) any {
	if s == "" {
		return nil
	}
	return s
}

func insertTurnMessage(ctx context.Context, tx pgx.Tx, sessionID, requestID uuid.UUID, role, content string, action *string, meta map[string]any) (*Message, error) {
	if meta == nil {
		meta = map[string]any{}
	}
	raw, err := json.Marshal(meta)
	if err != nil {
		return nil, err
	}
	var m Message
	var metaOut []byte
	err = tx.QueryRow(ctx, `
		INSERT INTO conversation_messages (session_id, request_id, role, content, action, message_metadata, created_at)
		VALUES ($1, $2, $3, $4, $5, $6::jsonb, clock_timestamp())
		RETURNING id, session_id, request_id, role, content, action, message_metadata, created_at`,
		sessionID, requestID, role, content, action, raw,
	).Scan(&m.ID, &m.SessionID, &m.RequestID, &m.Role, &m.Content, &m.Action, &metaOut, &m.CreatedAt)
	if err != nil {
		return nil, err
	}
	m.Metadata = decodeMessageMetadata(metaOut)
	return &m, nil
}

// LoadProcedureVersion loads a specific procedure_versions row (for confirm pin).
func (r *SessionRepo) LoadProcedureVersion(ctx context.Context, tx pgx.Tx, procedureID, versionID uuid.UUID) (*ProcedureDefinitionRow, error) {
	var row ProcedureDefinitionRow
	err := tx.QueryRow(ctx, `
		SELECT p.id, pv.id, p.procedure_code, p.name, pv.version, pv.definition
		FROM procedure_versions pv
		JOIN procedures p ON p.id = pv.procedure_id
		WHERE pv.procedure_id = $1 AND pv.id = $2`,
		procedureID, versionID,
	).Scan(&row.ProcedureID, &row.VersionID, &row.ProcedureCode, &row.Name, &row.Version, &row.Definition)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, ErrNotFound
	}
	if err != nil {
		return nil, err
	}
	return &row, nil
}

// FindIdempotency returns a COMPLETED turn envelope. In-flight PENDING claims
// are invisible here so the keyword-only path never replays a null body.
func (r *SessionRepo) FindIdempotency(ctx context.Context, tx pgx.Tx, sessionID, requestID uuid.UUID) (payloadHash string, responseJSON []byte, ok bool, err error) {
	err = tx.QueryRow(ctx, `
		SELECT payload_hash, response_json
		FROM turn_idempotency
		WHERE session_id = $1 AND request_id = $2 AND status = 'COMPLETE'`, sessionID, requestID,
	).Scan(&payloadHash, &responseJSON)
	if errors.Is(err, pgx.ErrNoRows) {
		return "", nil, false, nil
	}
	if err != nil {
		return "", nil, false, err
	}
	return payloadHash, responseJSON, true, nil
}

// FindIdempotencyState returns the claim row whether it is PENDING or COMPLETE.
func (r *SessionRepo) FindIdempotencyState(ctx context.Context, tx pgx.Tx, sessionID, requestID uuid.UUID) (IdempotencyState, bool, error) {
	var st IdempotencyState
	err := tx.QueryRow(ctx, `
		SELECT status, payload_hash, response_json, claim_expires_at
		FROM turn_idempotency
		WHERE session_id = $1 AND request_id = $2`, sessionID, requestID,
	).Scan(&st.Status, &st.PayloadHash, &st.ResponseJSON, &st.ExpiresAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return IdempotencyState{}, false, nil
	}
	if err != nil {
		return IdempotencyState{}, false, err
	}
	return st, true, nil
}

// ResolveExtractClaim atomically inserts a PENDING lease or reports that
// another request already owns this (session_id, request_id).
// The caller must commit this transaction before any network call.
// lease is the time after which a crashed owner's PENDING row may be stolen.
func (r *SessionRepo) ResolveExtractClaim(ctx context.Context, tx pgx.Tx, sessionID, requestID uuid.UUID, payloadHash string, lease time.Duration) (ClaimResult, error) {
	secs := int(lease.Seconds())
	if secs < 1 {
		secs = 1
	}
	var status, hash string
	var expires *time.Time
	err := tx.QueryRow(ctx, `
		SELECT status, payload_hash, claim_expires_at
		FROM turn_idempotency
		WHERE session_id = $1 AND request_id = $2
		FOR UPDATE`, sessionID, requestID,
	).Scan(&status, &hash, &expires)
	if errors.Is(err, pgx.ErrNoRows) {
		_, err = tx.Exec(ctx, `
			INSERT INTO turn_idempotency
				(session_id, request_id, payload_hash, response_json, status, claimed_at, claim_expires_at)
			VALUES ($1, $2, $3, NULL, 'PENDING', now(), now() + ($4 * interval '1 second'))`,
			sessionID, requestID, payloadHash, secs,
		)
		if err != nil {
			return 0, err
		}
		return ClaimOwned, nil
	}
	if err != nil {
		return 0, err
	}
	if hash != payloadHash {
		return ClaimConflict, nil
	}
	if status == IdempotencyComplete {
		return ClaimWait, nil
	}
	expired := expires == nil || !expires.After(time.Now())
	if !expired {
		return ClaimWait, nil
	}
	tag, err := tx.Exec(ctx, `
		UPDATE turn_idempotency
		SET claimed_at = now(),
		    claim_expires_at = now() + ($4 * interval '1 second')
		WHERE session_id = $1 AND request_id = $2
		  AND status = 'PENDING' AND payload_hash = $3
		  AND claim_expires_at <= now()`,
		sessionID, requestID, payloadHash, secs,
	)
	if err != nil {
		return 0, err
	}
	if tag.RowsAffected() == 1 {
		return ClaimOwned, nil
	}
	return ClaimWait, nil
}

// SaveIdempotency stores the full turn envelope for replay.
// A PENDING claim for the same hash is promoted to COMPLETE. A completed row
// with the same hash is left as-is. A different hash is a conflict.
func (r *SessionRepo) SaveIdempotency(ctx context.Context, tx pgx.Tx, sessionID, requestID uuid.UUID, payloadHash string, responseJSON []byte) error {
	tag, err := tx.Exec(ctx, `
		INSERT INTO turn_idempotency (session_id, request_id, payload_hash, response_json, status)
		VALUES ($1, $2, $3, $4::jsonb, 'COMPLETE')
		ON CONFLICT (session_id, request_id) DO UPDATE
		SET response_json = EXCLUDED.response_json,
		    status = 'COMPLETE',
		    claim_expires_at = NULL
		WHERE turn_idempotency.payload_hash = EXCLUDED.payload_hash
		  AND turn_idempotency.status = 'PENDING'`,
		sessionID, requestID, payloadHash, responseJSON,
	)
	if err != nil {
		return err
	}
	if tag.RowsAffected() > 0 {
		return nil
	}
	var status, hash string
	err = tx.QueryRow(ctx, `
		SELECT status, payload_hash
		FROM turn_idempotency
		WHERE session_id = $1 AND request_id = $2`, sessionID, requestID,
	).Scan(&status, &hash)
	if errors.Is(err, pgx.ErrNoRows) {
		return fmt.Errorf("idempotency row missing after save")
	}
	if err != nil {
		return err
	}
	if hash != payloadHash {
		return ErrIdempotencyConflict
	}
	if status == IdempotencyComplete {
		return nil
	}
	return fmt.Errorf("idempotency claim not completed")
}

func isUniqueViolation(err error) bool {
	if err == nil {
		return false
	}
	msg := err.Error()
	return strings.Contains(msg, "ux_conversation_messages_session_request_user") ||
		strings.Contains(msg, "duplicate key")
}
