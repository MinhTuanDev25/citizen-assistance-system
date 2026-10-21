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
	DraftStatusDraft     = "DRAFT"
	DraftStatusReviewed  = "REVIEWED"
	DraftStatusApproved  = "APPROVED"
	DraftStatusRejected  = "REJECTED"
	DraftStatusPublished = "PUBLISHED"
)

type ProcedureDraft struct {
	ID               uuid.UUID
	DocumentID       uuid.UUID
	ProcedureID      uuid.UUID
	DraftDefinition  json.RawMessage
	ValidationResult json.RawMessage
	Status           string
	CreatedBy        *uuid.UUID
	UpdatedBy        *uuid.UUID
	CreatedAt        time.Time
	UpdatedAt        time.Time
	ProcedureCode    string
	XaID             string
}

type DraftRepo struct {
	Pool *pgxpool.Pool
}

func (r *DraftRepo) Insert(ctx context.Context, row ProcedureDraft) (ProcedureDraft, error) {
	err := r.Pool.QueryRow(ctx, `
		INSERT INTO procedure_drafts (
			id, document_id, procedure_id, draft_definition, validation_result,
			status, created_by, updated_by
		) VALUES (
			COALESCE(NULLIF($1, '00000000-0000-0000-0000-000000000000'::uuid), gen_random_uuid()),
			$2, $3, $4, $5, $6, $7, $8
		)
		RETURNING id, document_id, procedure_id, draft_definition, validation_result,
			status, created_by, updated_by, created_at, updated_at`,
		row.ID, row.DocumentID, row.ProcedureID, row.DraftDefinition, row.ValidationResult,
		row.Status, row.CreatedBy, row.UpdatedBy,
	).Scan(
		&row.ID, &row.DocumentID, &row.ProcedureID, &row.DraftDefinition, &row.ValidationResult,
		&row.Status, &row.CreatedBy, &row.UpdatedBy, &row.CreatedAt, &row.UpdatedAt,
	)
	return row, err
}

func (r *DraftRepo) Get(ctx context.Context, xaID string, draftID uuid.UUID) (ProcedureDraft, error) {
	var row ProcedureDraft
	err := r.Pool.QueryRow(ctx, `
		SELECT d.id, d.document_id, d.procedure_id, d.draft_definition, d.validation_result,
			d.status, d.created_by, d.updated_by, d.created_at, d.updated_at,
			p.procedure_code, p.xa_id
		FROM procedure_drafts d
		JOIN procedures p ON p.id = d.procedure_id
		WHERE d.id = $1 AND p.xa_id = $2`,
		draftID, xaID,
	).Scan(
		&row.ID, &row.DocumentID, &row.ProcedureID, &row.DraftDefinition, &row.ValidationResult,
		&row.Status, &row.CreatedBy, &row.UpdatedBy, &row.CreatedAt, &row.UpdatedAt,
		&row.ProcedureCode, &row.XaID,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return ProcedureDraft{}, ErrNotFound
	}
	return row, err
}

func (r *DraftRepo) ListByProcedure(ctx context.Context, xaID string, procedureID uuid.UUID) ([]ProcedureDraft, error) {
	rows, err := r.Pool.Query(ctx, `
		SELECT d.id, d.document_id, d.procedure_id, d.draft_definition, d.validation_result,
			d.status, d.created_by, d.updated_by, d.created_at, d.updated_at,
			p.procedure_code, p.xa_id
		FROM procedure_drafts d
		JOIN procedures p ON p.id = d.procedure_id
		WHERE d.procedure_id = $1 AND p.xa_id = $2
		ORDER BY d.created_at DESC`,
		procedureID, xaID,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := make([]ProcedureDraft, 0)
	for rows.Next() {
		var row ProcedureDraft
		if err := rows.Scan(
			&row.ID, &row.DocumentID, &row.ProcedureID, &row.DraftDefinition, &row.ValidationResult,
			&row.Status, &row.CreatedBy, &row.UpdatedBy, &row.CreatedAt, &row.UpdatedAt,
			&row.ProcedureCode, &row.XaID,
		); err != nil {
			return nil, err
		}
		out = append(out, row)
	}
	return out, rows.Err()
}

func (r *DraftRepo) UpdateDefinition(ctx context.Context, xaID string, draftID uuid.UUID, actor uuid.UUID, definition, validation json.RawMessage, expectedUpdatedAt time.Time) (ProcedureDraft, error) {
	var row ProcedureDraft
	err := r.Pool.QueryRow(ctx, `
		UPDATE procedure_drafts d
		SET draft_definition = $3,
			validation_result = $4,
			updated_by = $5,
			updated_at = now()
		FROM procedures p
		WHERE d.id = $1
		  AND d.procedure_id = p.id
		  AND p.xa_id = $2
		  AND d.status = 'DRAFT'
		  AND d.updated_at = $6
		RETURNING d.id, d.document_id, d.procedure_id, d.draft_definition, d.validation_result,
			d.status, d.created_by, d.updated_by, d.created_at, d.updated_at,
			p.procedure_code, p.xa_id`,
		draftID, xaID, definition, validation, actor, expectedUpdatedAt,
	).Scan(
		&row.ID, &row.DocumentID, &row.ProcedureID, &row.DraftDefinition, &row.ValidationResult,
		&row.Status, &row.CreatedBy, &row.UpdatedBy, &row.CreatedAt, &row.UpdatedAt,
		&row.ProcedureCode, &row.XaID,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return ProcedureDraft{}, ErrNotFound
	}
	return row, err
}

func (r *DraftRepo) ListAll(ctx context.Context, xaID string, procedureID *uuid.UUID) ([]ProcedureDraft, error) {
	query := `
		SELECT d.id, d.document_id, d.procedure_id, d.draft_definition, d.validation_result,
			d.status, d.created_by, d.updated_by, d.created_at, d.updated_at,
			p.procedure_code, p.xa_id
		FROM procedure_drafts d
		JOIN procedures p ON p.id = d.procedure_id
		WHERE p.xa_id = $1`
	args := []any{xaID}
	if procedureID != nil {
		query += ` AND d.procedure_id = $2`
		args = append(args, *procedureID)
	}
	query += ` ORDER BY d.updated_at DESC`
	rows, err := r.Pool.Query(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := make([]ProcedureDraft, 0)
	for rows.Next() {
		var row ProcedureDraft
		if err := rows.Scan(
			&row.ID, &row.DocumentID, &row.ProcedureID, &row.DraftDefinition, &row.ValidationResult,
			&row.Status, &row.CreatedBy, &row.UpdatedBy, &row.CreatedAt, &row.UpdatedAt,
			&row.ProcedureCode, &row.XaID,
		); err != nil {
			return nil, err
		}
		out = append(out, row)
	}
	return out, rows.Err()
}

func (r *DraftRepo) UpdateStatus(ctx context.Context, xaID string, draftID uuid.UUID, actor uuid.UUID, status string, definition, validation json.RawMessage, expectedUpdatedAt time.Time, allowedFrom []string) (ProcedureDraft, error) {
	var row ProcedureDraft
	err := r.Pool.QueryRow(ctx, `
		UPDATE procedure_drafts d
		SET draft_definition = $3,
			validation_result = $4,
			status = $5,
			updated_by = $6,
			updated_at = now()
		FROM procedures p
		WHERE d.id = $1
		  AND d.procedure_id = p.id
		  AND p.xa_id = $2
		  AND d.status = ANY($7)
		  AND d.updated_at = $8
		RETURNING d.id, d.document_id, d.procedure_id, d.draft_definition, d.validation_result,
			d.status, d.created_by, d.updated_by, d.created_at, d.updated_at,
			p.procedure_code, p.xa_id`,
		draftID, xaID, definition, validation, status, actor, allowedFrom, expectedUpdatedAt,
	).Scan(
		&row.ID, &row.DocumentID, &row.ProcedureID, &row.DraftDefinition, &row.ValidationResult,
		&row.Status, &row.CreatedBy, &row.UpdatedBy, &row.CreatedAt, &row.UpdatedAt,
		&row.ProcedureCode, &row.XaID,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return ProcedureDraft{}, ErrNotFound
	}
	return row, err
}
