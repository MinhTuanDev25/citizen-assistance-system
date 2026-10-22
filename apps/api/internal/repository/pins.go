package repository

import (
	"context"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
)

type SourcePin struct {
	DocumentID         uuid.UUID
	GenerationID       uuid.UUID
	ProcedureID        uuid.UUID
	ProcedureVersionID uuid.UUID
	Title              string
	DocumentNumber     string
	ContentSHA256      string
	SourceSHA256       string
}

func (r *SessionRepo) LoadSourcePins(ctx context.Context, tx pgx.Tx, sessionID, versionID uuid.UUID) ([]SourcePin, error) {
	rows, err := tx.Query(ctx, `
		SELECT document_id, generation_id, procedure_id, procedure_version_id,
		       document_title, COALESCE(document_number, ''), COALESCE(content_sha256, ''), COALESCE(source_sha256, '')
		FROM session_source_pins
		WHERE session_id = $1 AND procedure_version_id = $2
		ORDER BY created_at, document_id`, sessionID, versionID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := make([]SourcePin, 0)
	for rows.Next() {
		var pin SourcePin
		if err := rows.Scan(
			&pin.DocumentID, &pin.GenerationID, &pin.ProcedureID, &pin.ProcedureVersionID,
			&pin.Title, &pin.DocumentNumber, &pin.ContentSHA256, &pin.SourceSHA256,
		); err != nil {
			return nil, err
		}
		out = append(out, pin)
	}
	return out, rows.Err()
}

// EnsureSourcePins freezes READY links for the pinned version once.
// An empty freeze is still a freeze: later activate/reindex/unlink cannot
// attach a different generation.
func (r *SessionRepo) EnsureSourcePins(ctx context.Context, tx pgx.Tx, sessionID uuid.UUID, xaID string, procedureID, versionID uuid.UUID) ([]SourcePin, error) {
	if _, err := tx.Exec(ctx, `
		INSERT INTO session_source_pin_sets (session_id, procedure_version_id, procedure_id, xa_id)
		VALUES ($1, $2, $3, $4)
		ON CONFLICT (session_id, procedure_version_id) DO NOTHING`,
		sessionID, versionID, procedureID, xaID,
	); err != nil {
		return nil, err
	}
	var materialized bool
	if err := tx.QueryRow(ctx, `
		SELECT materialized FROM session_source_pin_sets
		WHERE session_id = $1 AND procedure_version_id = $2
		FOR UPDATE`, sessionID, versionID).Scan(&materialized); err != nil {
		return nil, err
	}
	if !materialized {
		if _, err := tx.Exec(ctx, `
			INSERT INTO session_source_pins (
				session_id, xa_id, procedure_id, procedure_version_id,
				document_id, generation_id, document_title, document_number,
				content_sha256, source_sha256
			)
			SELECT $1, $2, $3, $4,
			       link.document_id, link.active_generation_id, d.title, d.document_number,
			       g.content_sha256, g.source_sha256
			FROM procedure_version_documents link
			JOIN documents d ON d.id = link.document_id
			JOIN document_index_generations g ON g.id = link.active_generation_id
			WHERE link.procedure_version_id = $4
			  AND link.xa_id = $2
			  AND link.unlinked_at IS NULL
			  AND link.index_status = 'READY'
			  AND link.active_generation_id IS NOT NULL
			ON CONFLICT (session_id, procedure_version_id, document_id) DO NOTHING`,
			sessionID, xaID, procedureID, versionID,
		); err != nil {
			return nil, err
		}
		if _, err := tx.Exec(ctx, `
			UPDATE session_source_pin_sets
			SET materialized = TRUE
			WHERE session_id = $1 AND procedure_version_id = $2`,
			sessionID, versionID,
		); err != nil {
			return nil, err
		}
	}
	return r.LoadSourcePins(ctx, tx, sessionID, versionID)
}

func (r *SessionRepo) LoadPinnedChunks(ctx context.Context, tx pgx.Tx, xaID string, versionID uuid.UUID, pins []SourcePin, limit int) ([]PinnedChunk, error) {
	if len(pins) == 0 {
		return nil, nil
	}
	docIDs := make([]uuid.UUID, len(pins))
	genIDs := make([]uuid.UUID, len(pins))
	for i, pin := range pins {
		docIDs[i] = pin.DocumentID
		genIDs[i] = pin.GenerationID
	}
	rows, err := tx.Query(ctx, `
		SELECT c.id, c.document_id, c.generation_id, d.title, COALESCE(d.document_number, ''),
		       c.content, c.page_start, c.page_end, COALESCE(c.text_sha256, '')
		FROM knowledge_chunks c
		JOIN documents d ON d.id = c.document_id
		JOIN unnest($3::uuid[], $4::uuid[]) AS pin(document_id, generation_id)
		  ON pin.document_id = c.document_id AND pin.generation_id = c.generation_id
		WHERE c.xa_id = $1 AND c.procedure_version_id = $2
		ORDER BY c.chunk_index
		LIMIT $5`, xaID, versionID, docIDs, genIDs, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := make([]PinnedChunk, 0)
	for rows.Next() {
		var row PinnedChunk
		if err := rows.Scan(
			&row.ChunkID, &row.DocumentID, &row.GenerationID, &row.Title, &row.DocumentNumber,
			&row.Content, &row.PageStart, &row.PageEnd, &row.TextSHA256,
		); err != nil {
			return nil, err
		}
		applyPinSnapshot(&row, pins)
		out = append(out, row)
	}
	return out, rows.Err()
}

func applyPinSnapshot(row *PinnedChunk, pins []SourcePin) {
	if row == nil {
		return
	}
	for _, pin := range pins {
		if pin.DocumentID != row.DocumentID {
			continue
		}
		if pin.Title != "" {
			row.Title = pin.Title
		}
		if pin.DocumentNumber != "" {
			row.DocumentNumber = pin.DocumentNumber
		}
		return
	}
}

type PinnedChunk struct {
	ChunkID        uuid.UUID
	DocumentID     uuid.UUID
	GenerationID   uuid.UUID
	Title          string
	DocumentNumber string
	Content        string
	PageStart      int
	PageEnd        int
	TextSHA256     string
	Score          int
}
