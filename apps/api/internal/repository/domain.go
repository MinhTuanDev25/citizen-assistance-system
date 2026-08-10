package repository

import (
	"context"
	"errors"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
)

var (
	ErrNotFound      = errors.New("not_found")
	ErrConflict      = errors.New("conflict")
	ErrFKRestricted  = errors.New("fk_restricted")
)

type Domain struct {
	ID          string    `json:"id"`
	Name        string    `json:"name"`
	Description *string   `json:"description,omitempty"`
	SortOrder   int       `json:"sort_order"`
	IsActive    bool      `json:"is_active"`
	CreatedAt   time.Time `json:"created_at"`
}

type DomainCreate struct {
	ID          string  `json:"id" binding:"required"`
	Name        string  `json:"name" binding:"required"`
	Description *string `json:"description"`
	SortOrder   *int    `json:"sort_order"`
	IsActive    *bool   `json:"is_active"`
}

type DomainUpdate struct {
	Name        *string `json:"name"`
	Description *string `json:"description"`
	SortOrder   *int    `json:"sort_order"`
	IsActive    *bool   `json:"is_active"`
}

type DomainRepo struct {
	Pool *pgxpool.Pool
}

func (r *DomainRepo) List(ctx context.Context, activeOnly bool) ([]Domain, error) {
	q := `
		SELECT id, name, description, sort_order, is_active, created_at
		FROM domains`
	if activeOnly {
		q += ` WHERE is_active = true`
	}
	q += ` ORDER BY sort_order, id`

	rows, err := r.Pool.Query(ctx, q)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := make([]Domain, 0)
	for rows.Next() {
		var d Domain
		if err := rows.Scan(&d.ID, &d.Name, &d.Description, &d.SortOrder, &d.IsActive, &d.CreatedAt); err != nil {
			return nil, err
		}
		out = append(out, d)
	}
	return out, rows.Err()
}

func (r *DomainRepo) GetByID(ctx context.Context, id string) (*Domain, error) {
	var d Domain
	err := r.Pool.QueryRow(ctx, `
		SELECT id, name, description, sort_order, is_active, created_at
		FROM domains WHERE id = $1`, id,
	).Scan(&d.ID, &d.Name, &d.Description, &d.SortOrder, &d.IsActive, &d.CreatedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, ErrNotFound
	}
	if err != nil {
		return nil, err
	}
	return &d, nil
}

func (r *DomainRepo) Create(ctx context.Context, in DomainCreate) (*Domain, error) {
	sortOrder := 0
	if in.SortOrder != nil {
		sortOrder = *in.SortOrder
	}
	active := true
	if in.IsActive != nil {
		active = *in.IsActive
	}

	var d Domain
	err := r.Pool.QueryRow(ctx, `
		INSERT INTO domains (id, name, description, sort_order, is_active)
		VALUES ($1, $2, $3, $4, $5)
		RETURNING id, name, description, sort_order, is_active, created_at`,
		in.ID, in.Name, in.Description, sortOrder, active,
	).Scan(&d.ID, &d.Name, &d.Description, &d.SortOrder, &d.IsActive, &d.CreatedAt)
	if err != nil {
		var pgErr *pgconn.PgError
		if errors.As(err, &pgErr) && pgErr.Code == "23505" {
			return nil, ErrConflict
		}
		return nil, err
	}
	return &d, nil
}

func (r *DomainRepo) Update(ctx context.Context, id string, in DomainUpdate) (*Domain, error) {
	cur, err := r.GetByID(ctx, id)
	if err != nil {
		return nil, err
	}

	name := cur.Name
	if in.Name != nil {
		name = *in.Name
	}
	desc := cur.Description
	if in.Description != nil {
		desc = in.Description
	}
	sortOrder := cur.SortOrder
	if in.SortOrder != nil {
		sortOrder = *in.SortOrder
	}
	active := cur.IsActive
	if in.IsActive != nil {
		active = *in.IsActive
	}

	var d Domain
	err = r.Pool.QueryRow(ctx, `
		UPDATE domains
		SET name = $2, description = $3, sort_order = $4, is_active = $5
		WHERE id = $1
		RETURNING id, name, description, sort_order, is_active, created_at`,
		id, name, desc, sortOrder, active,
	).Scan(&d.ID, &d.Name, &d.Description, &d.SortOrder, &d.IsActive, &d.CreatedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, ErrNotFound
	}
	if err != nil {
		return nil, err
	}
	return &d, nil
}

// SoftDelete sets is_active = false (safe with FK RESTRICT on procedures/documents).
func (r *DomainRepo) SoftDelete(ctx context.Context, id string) (*Domain, error) {
	var d Domain
	err := r.Pool.QueryRow(ctx, `
		UPDATE domains SET is_active = false
		WHERE id = $1
		RETURNING id, name, description, sort_order, is_active, created_at`,
		id,
	).Scan(&d.ID, &d.Name, &d.Description, &d.SortOrder, &d.IsActive, &d.CreatedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, ErrNotFound
	}
	if err != nil {
		return nil, err
	}
	return &d, nil
}

// HardDelete removes the row; fails with ErrFKRestricted if referenced.
func (r *DomainRepo) HardDelete(ctx context.Context, id string) error {
	tag, err := r.Pool.Exec(ctx, `DELETE FROM domains WHERE id = $1`, id)
	if err != nil {
		var pgErr *pgconn.PgError
		if errors.As(err, &pgErr) && pgErr.Code == "23503" {
			return ErrFKRestricted
		}
		return err
	}
	if tag.RowsAffected() == 0 {
		return ErrNotFound
	}
	return nil
}
