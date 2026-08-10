package repository

import (
	"context"
	"errors"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

type Commune struct {
	ID          string    `json:"id"`
	Name        string    `json:"name"`
	Description *string   `json:"description,omitempty"`
	IsActive    bool      `json:"is_active"`
	CreatedAt   time.Time `json:"created_at"`
}

type CommuneRepo struct {
	Pool *pgxpool.Pool
}

func (r *CommuneRepo) List(ctx context.Context, activeOnly bool) ([]Commune, error) {
	q := `
		SELECT id, name, description, is_active, created_at
		FROM communes`
	if activeOnly {
		q += ` WHERE is_active = true`
	}
	q += ` ORDER BY id`

	rows, err := r.Pool.Query(ctx, q)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := make([]Commune, 0)
	for rows.Next() {
		var c Commune
		if err := rows.Scan(&c.ID, &c.Name, &c.Description, &c.IsActive, &c.CreatedAt); err != nil {
			return nil, err
		}
		out = append(out, c)
	}
	return out, rows.Err()
}

func (r *CommuneRepo) GetByID(ctx context.Context, id string) (*Commune, error) {
	var c Commune
	err := r.Pool.QueryRow(ctx, `
		SELECT id, name, description, is_active, created_at
		FROM communes WHERE id = $1`, id,
	).Scan(&c.ID, &c.Name, &c.Description, &c.IsActive, &c.CreatedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return &c, nil
}
