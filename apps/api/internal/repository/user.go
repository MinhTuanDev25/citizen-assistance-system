package repository

import (
	"context"
	"errors"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

type User struct {
	ID           uuid.UUID `json:"id"`
	Role         string    `json:"role"`
	FullName     string    `json:"full_name"`
	Email        *string   `json:"email,omitempty"`
	Phone        *string   `json:"phone,omitempty"`
	PasswordHash *string   `json:"-"`
	CreatedAt    time.Time `json:"created_at"`
	UpdatedAt    time.Time `json:"updated_at"`
}

type UserPublic struct {
	ID        uuid.UUID `json:"id"`
	Role      string    `json:"role"`
	FullName  string    `json:"full_name"`
	Email     *string   `json:"email,omitempty"`
	Phone     *string   `json:"phone,omitempty"`
	CreatedAt time.Time `json:"created_at"`
}

func (u User) Public() UserPublic {
	return UserPublic{
		ID:        u.ID,
		Role:      u.Role,
		FullName:  u.FullName,
		Email:     u.Email,
		Phone:     u.Phone,
		CreatedAt: u.CreatedAt,
	}
}

type UserRepo struct {
	Pool *pgxpool.Pool
}

func (r *UserRepo) GetByID(ctx context.Context, id uuid.UUID) (*User, error) {
	return r.scanOne(ctx, `
		SELECT id, role, full_name, email, phone, password_hash, created_at, updated_at
		FROM users WHERE id = $1`, id)
}

func (r *UserRepo) GetByEmail(ctx context.Context, email string) (*User, error) {
	email = strings.TrimSpace(strings.ToLower(email))
	return r.scanOne(ctx, `
		SELECT id, role, full_name, email, phone, password_hash, created_at, updated_at
		FROM users WHERE lower(email) = $1`, email)
}

func (r *UserRepo) CreateCitizen(ctx context.Context, fullName, email, passwordHash string) (*User, error) {
	email = strings.TrimSpace(strings.ToLower(email))
	fullName = strings.TrimSpace(fullName)
	var u User
	err := r.Pool.QueryRow(ctx, `
		INSERT INTO users (role, full_name, email, password_hash)
		VALUES ('CITIZEN', $1, $2, $3)
		RETURNING id, role, full_name, email, phone, password_hash, created_at, updated_at`,
		fullName, email, passwordHash,
	).Scan(&u.ID, &u.Role, &u.FullName, &u.Email, &u.Phone, &u.PasswordHash, &u.CreatedAt, &u.UpdatedAt)
	if err != nil {
		return nil, err
	}
	return &u, nil
}

func (r *UserRepo) scanOne(ctx context.Context, q string, args ...any) (*User, error) {
	var u User
	err := r.Pool.QueryRow(ctx, q, args...).Scan(
		&u.ID, &u.Role, &u.FullName, &u.Email, &u.Phone, &u.PasswordHash, &u.CreatedAt, &u.UpdatedAt,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return &u, nil
}
