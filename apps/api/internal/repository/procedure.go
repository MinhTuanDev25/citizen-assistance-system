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

// ProcedureListItem is a lightweight row for catalog listing.
type ProcedureListItem struct {
	ID                  uuid.UUID `json:"id"`
	ProcedureCode       string    `json:"procedure_code"`
	Name                string    `json:"name"`
	DomainID            string    `json:"domain_id"`
	DomainName          string    `json:"domain_name"`
	DomainSortOrder     int       `json:"-"`
	XaID                string    `json:"xa_id"`
	ActiveVersionID     uuid.UUID `json:"active_version_id"`
	ActiveVersion       string    `json:"active_version"`
	ActiveVersionStatus string    `json:"active_version_status"`
}

// Procedure is a full procedure row (no definition blob).
type Procedure struct {
	ID              uuid.UUID  `json:"id"`
	ProcedureCode   string     `json:"procedure_code"`
	DomainID        string     `json:"domain_id"`
	Name            string     `json:"name"`
	XaID            string     `json:"xa_id"`
	ActiveVersionID *uuid.UUID `json:"active_version_id,omitempty"`
	CreatedAt       time.Time  `json:"created_at"`
	UpdatedAt       time.Time  `json:"updated_at"`
}

// ActiveVersion carries the ACTIVE procedure_versions row + definition JSON.
type ActiveVersion struct {
	ID            uuid.UUID       `json:"id"`
	ProcedureID   uuid.UUID       `json:"procedure_id"`
	Version       string          `json:"version"`
	Status        string          `json:"status"`
	Definition    json.RawMessage `json:"definition"`
	CreatedAt     time.Time       `json:"created_at"`
	ApprovedAt    *time.Time      `json:"approved_at,omitempty"`
	ProcedureCode string          `json:"procedure_code"`
	ProcedureName string          `json:"procedure_name"`
	XaID          string          `json:"xa_id"`
	DomainID      string          `json:"domain_id"`
}

type ProcedureRepo struct {
	Pool *pgxpool.Pool
}

// ListActive returns procedures for a commune that have an ACTIVE version pointer.
func (r *ProcedureRepo) ListActive(ctx context.Context, xaID, domainID string) ([]ProcedureListItem, error) {
	q := `
		SELECT
			p.id,
			p.procedure_code,
			p.name,
			p.domain_id,
			d.name AS domain_name,
			d.sort_order AS domain_sort_order,
			p.xa_id,
			p.active_version_id,
			pv.version,
			pv.status
		FROM procedures p
		JOIN domains d ON d.id = p.domain_id
		JOIN procedure_versions pv ON pv.id = p.active_version_id AND pv.procedure_id = p.id
		WHERE p.xa_id = $1
		  AND p.active_version_id IS NOT NULL
		  AND pv.status = 'ACTIVE'`
	args := []any{xaID}
	if domainID != "" {
		q += ` AND p.domain_id = $2`
		args = append(args, domainID)
	}
	q += ` ORDER BY d.sort_order, p.procedure_code`

	rows, err := r.Pool.Query(ctx, q, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := make([]ProcedureListItem, 0)
	for rows.Next() {
		var item ProcedureListItem
		if err := rows.Scan(
			&item.ID,
			&item.ProcedureCode,
			&item.Name,
			&item.DomainID,
			&item.DomainName,
			&item.DomainSortOrder,
			&item.XaID,
			&item.ActiveVersionID,
			&item.ActiveVersion,
			&item.ActiveVersionStatus,
		); err != nil {
			return nil, err
		}
		out = append(out, item)
	}
	return out, rows.Err()
}

// GetByCode resolves (xa_id, procedure_code).
func (r *ProcedureRepo) GetByCode(ctx context.Context, xaID, code string) (*Procedure, error) {
	var p Procedure
	err := r.Pool.QueryRow(ctx, `
		SELECT id, procedure_code, domain_id, name, xa_id, active_version_id, created_at, updated_at
		FROM procedures
		WHERE xa_id = $1 AND procedure_code = $2`,
		xaID, code,
	).Scan(
		&p.ID, &p.ProcedureCode, &p.DomainID, &p.Name, &p.XaID,
		&p.ActiveVersionID, &p.CreatedAt, &p.UpdatedAt,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, ErrNotFound
	}
	if err != nil {
		return nil, err
	}
	return &p, nil
}

// GetActiveVersion returns ACTIVE version + definition for procedure id.
func (r *ProcedureRepo) GetActiveVersion(ctx context.Context, procedureID uuid.UUID) (*ActiveVersion, error) {
	var v ActiveVersion
	err := r.Pool.QueryRow(ctx, `
		SELECT
			pv.id,
			pv.procedure_id,
			pv.version,
			pv.status,
			pv.definition,
			pv.created_at,
			pv.approved_at,
			p.procedure_code,
			p.name,
			p.xa_id,
			p.domain_id
		FROM procedures p
		JOIN procedure_versions pv
		  ON pv.id = p.active_version_id
		 AND pv.procedure_id = p.id
		WHERE p.id = $1
		  AND p.active_version_id IS NOT NULL
		  AND pv.status = 'ACTIVE'`,
		procedureID,
	).Scan(
		&v.ID,
		&v.ProcedureID,
		&v.Version,
		&v.Status,
		&v.Definition,
		&v.CreatedAt,
		&v.ApprovedAt,
		&v.ProcedureCode,
		&v.ProcedureName,
		&v.XaID,
		&v.DomainID,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, ErrNotFound
	}
	if err != nil {
		return nil, err
	}
	return &v, nil
}
