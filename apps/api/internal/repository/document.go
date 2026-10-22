package repository

import (
	"context"
	"errors"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/document"
	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
)

type DocumentRepo struct {
	Pool *pgxpool.Pool
}

func (r *DocumentRepo) DomainActive(ctx context.Context, id string) error {
	var active bool
	err := r.Pool.QueryRow(ctx, `SELECT is_active FROM domains WHERE id = $1`, id).Scan(&active)
	if errors.Is(err, pgx.ErrNoRows) || (err == nil && !active) {
		return document.ErrDomain
	}
	return err
}

func (r *DocumentRepo) Insert(ctx context.Context, doc document.Document, requestID uuid.UUID) error {
	tx, err := r.Pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx)

	_, err = tx.Exec(ctx, `
		INSERT INTO documents (
			id, xa_id, domain_id, title, document_number, issuer, filename,
			storage_uri, checksum, mime_type, file_size_bytes,
			effective_date, expire_date, issued_date,
			processing_status, validity_status, uploaded_by
		) VALUES (
			$1,$2,$3,$4,$5,$6,$7,
			$8,$9,$10,$11,
			$12::date,$13::date,$14::date,
			$15,$16,$17
		)`,
		doc.ID, doc.XAID, doc.DomainID, doc.Title, doc.DocumentNumber, doc.Issuer, doc.Filename,
		doc.StorageURI, doc.Checksum, doc.MimeType, doc.FileSizeBytes,
		nullDate(doc.EffectiveDate), nullDate(doc.ExpireDate), nullDate(doc.IssuedDate),
		doc.ProcessingStatus, doc.ValidityStatus, doc.UploadedBy,
	)
	if err != nil {
		var pgErr *pgconn.PgError
		if errors.As(err, &pgErr) && pgErr.Code == "23505" {
			return document.ErrDuplicate
		}
		return err
	}
	_, err = tx.Exec(ctx, `
		INSERT INTO audit_logs (request_id, actor_user_id, action, entity_type, entity_id, payload)
		VALUES ($1, $2, 'DOCUMENT_UPLOADED', 'document', $3, jsonb_build_object(
			'checksum', $4::text,
			'file_size_bytes', $5::bigint,
			'mime_type', $6::text,
			'filename', $7::text,
			'title', $8::text,
			'domain_id', $9::text,
			'xa_id', $10::text
		))`,
		requestID, doc.UploadedBy, doc.ID.String(),
		doc.Checksum, doc.FileSizeBytes, doc.MimeType, doc.Filename, doc.Title, doc.DomainID, doc.XAID,
	)
	if err != nil {
		return err
	}
	return tx.Commit(ctx)
}

func (r *DocumentRepo) Get(ctx context.Context, xaID string, id uuid.UUID) (document.Document, error) {
	row := r.Pool.QueryRow(ctx, documentSelect+` WHERE id = $1 AND xa_id = $2`, id, xaID)
	doc, err := scanDocument(row)
	if errors.Is(err, pgx.ErrNoRows) {
		return document.Document{}, document.ErrNotFound
	}
	return doc, err
}

func (r *DocumentRepo) List(ctx context.Context, f document.ListFilter) (document.ListResult, error) {
	where := ` WHERE xa_id = $1
		AND ($2 = '' OR domain_id = $2)
		AND ($3 = '' OR processing_status = $3)
		AND ($4 = '' OR validity_status = $4)`
	args := []any{f.XAID, f.DomainID, f.ProcessingStatus, f.ValidityStatus}
	var count int
	if err := r.Pool.QueryRow(ctx, `SELECT count(*) FROM documents`+where, args...).Scan(&count); err != nil {
		return document.ListResult{}, err
	}
	rows, err := r.Pool.Query(ctx, documentSelect+where+` ORDER BY created_at DESC LIMIT $5 OFFSET $6`,
		append(args, f.Limit, f.Offset)...)
	if err != nil {
		return document.ListResult{}, err
	}
	defer rows.Close()
	items := make([]document.Document, 0)
	for rows.Next() {
		doc, err := scanDocument(rows)
		if err != nil {
			return document.ListResult{}, err
		}
		items = append(items, doc)
	}
	if err := rows.Err(); err != nil {
		return document.ListResult{}, err
	}
	return document.ListResult{Items: items, Count: count, Limit: f.Limit, Offset: f.Offset}, nil
}

const documentSelect = `
	SELECT id, xa_id, domain_id, title, document_number, issuer, filename,
		storage_uri, checksum, mime_type, file_size_bytes,
		to_char(effective_date, 'YYYY-MM-DD'),
		to_char(expire_date, 'YYYY-MM-DD'),
		to_char(issued_date, 'YYYY-MM-DD'),
		processing_status, validity_status, uploaded_by, created_at, updated_at
	FROM documents`

type scannable interface {
	Scan(dest ...any) error
}

func scanDocument(row scannable) (document.Document, error) {
	var doc document.Document
	var domainID *string
	var effective, expire, issued *string
	err := row.Scan(
		&doc.ID, &doc.XAID, &domainID, &doc.Title, &doc.DocumentNumber, &doc.Issuer, &doc.Filename,
		&doc.StorageURI, &doc.Checksum, &doc.MimeType, &doc.FileSizeBytes,
		&effective, &expire, &issued,
		&doc.ProcessingStatus, &doc.ValidityStatus, &doc.UploadedBy, &doc.CreatedAt, &doc.UpdatedAt,
	)
	if err != nil {
		return document.Document{}, err
	}
	if domainID != nil {
		doc.DomainID = *domainID
	}
	doc.EffectiveDate = effective
	doc.ExpireDate = expire
	doc.IssuedDate = issued
	return doc, nil
}

func nullDate(v *string) any {
	if v == nil || *v == "" {
		return nil
	}
	return *v
}
