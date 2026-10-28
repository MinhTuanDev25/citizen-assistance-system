package activation

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
)

type Store struct {
	Pool *pgxpool.Pool
}

func (s *Store) begin(ctx context.Context) (pgx.Tx, error) {
	return s.Pool.Begin(ctx)
}

func hashPayload(parts ...string) string {
	h := sha256.New()
	for _, p := range parts {
		_, _ = h.Write([]byte(p))
		_, _ = h.Write([]byte{0})
	}
	return hex.EncodeToString(h.Sum(nil))
}

func claimIdempotency(ctx context.Context, tx pgx.Tx, xaID string, requestID uuid.UUID, action, hash string) ([]byte, bool, error) {
	tag, err := tx.Exec(ctx, `
		INSERT INTO admin_write_idempotency (xa_id, request_id, action, payload_hash, status_code, response_json)
		VALUES ($1,$2,$3,$4,0,'{}'::jsonb)
		ON CONFLICT (xa_id, request_id, action) DO NOTHING`, xaID, requestID, action, hash)
	if err != nil {
		return nil, false, err
	}
	if tag.RowsAffected() == 1 {
		return nil, false, nil
	}
	var got string
	var raw []byte
	err = tx.QueryRow(ctx, `
		SELECT payload_hash, response_json
		FROM admin_write_idempotency
		WHERE xa_id = $1 AND request_id = $2 AND action = $3
		FOR UPDATE`, xaID, requestID, action).Scan(&got, &raw)
	if err != nil {
		return nil, false, err
	}
	if got != hash {
		return nil, false, ErrIdempotency
	}
	return raw, true, nil
}

func storeIdempotency(ctx context.Context, tx pgx.Tx, xaID string, requestID uuid.UUID, action, hash string, status int, body []byte) error {
	tag, err := tx.Exec(ctx, `
		UPDATE admin_write_idempotency
		SET payload_hash = $4, status_code = $5, response_json = $6::jsonb
		WHERE xa_id = $1 AND request_id = $2 AND action = $3`, xaID, requestID, action, hash, status, body)
	if err != nil {
		return err
	}
	if tag.RowsAffected() != 1 {
		return errors.New("idempotency claim missing")
	}
	return nil
}

func lockProcedure(ctx context.Context, tx pgx.Tx, xaID string, procedureID uuid.UUID) (uuid.UUID, string, string, string, *uuid.UUID, error) {
	var id uuid.UUID
	var code, name, domain string
	var active *uuid.UUID
	err := tx.QueryRow(ctx, `
		SELECT id, procedure_code, name, domain_id, active_version_id
		FROM procedures
		WHERE id = $1 AND xa_id = $2
		FOR UPDATE`, procedureID, xaID).Scan(&id, &code, &name, &domain, &active)
	if errors.Is(err, pgx.ErrNoRows) {
		return uuid.Nil, "", "", "", nil, ErrNotFound
	}
	return id, code, name, domain, active, err
}

func loadLiveActive(ctx context.Context, tx pgx.Tx, procedureID uuid.UUID) (*uuid.UUID, error) {
	var id uuid.UUID
	err := tx.QueryRow(ctx, `
		SELECT id FROM procedure_versions
		WHERE procedure_id = $1 AND status = 'ACTIVE'
		FOR UPDATE`, procedureID).Scan(&id)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return &id, nil
}

func loadVersion(ctx context.Context, tx pgx.Tx, xaID string, procedureID, versionID uuid.UUID) (versionRow, error) {
	var row versionRow
	err := tx.QueryRow(ctx, `
		SELECT pv.id, pv.procedure_id, pv.version, pv.status, pv.definition,
		       pv.created_by, pv.approved_by, pv.created_at, pv.approved_at,
		       pv.activated_at, pv.activated_by,
		       p.xa_id, p.domain_id, p.procedure_code, p.name
		FROM procedure_versions pv
		JOIN procedures p ON p.id = pv.procedure_id
		WHERE pv.id = $1 AND pv.procedure_id = $2 AND p.xa_id = $3
		FOR UPDATE OF pv`, versionID, procedureID, xaID).Scan(
		&row.ID, &row.ProcedureID, &row.Version, &row.Status, &row.Definition,
		&row.CreatedBy, &row.ApprovedBy, &row.CreatedAt, &row.ApprovedAt,
		&row.ActivatedAt, &row.ActivatedBy,
		&row.XaID, &row.DomainID, &row.ProcedureCode, &row.ProcedureName,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return versionRow{}, ErrNotFound
	}
	return row, err
}

func loadSources(ctx context.Context, tx pgx.Tx, versionID uuid.UUID) ([]sourceRow, error) {
	rows, err := tx.Query(ctx, `
		SELECT link.document_id, d.title, link.relationship_type, link.index_status,
		       d.validity_status, link.unlinked_at IS NOT NULL,
		       link.xa_id, link.domain_id, link.procedure_id, link.procedure_version_id,
		       link.active_generation_id,
		       COALESCE(gen.status, ''),
		       COALESCE(gen.chunk_count, 0),
		       COALESCE(gen.vector_count, 0),
		       COALESCE(gen.vector_dimension, 0),
		       COALESCE(gen.embedding_model_id, ''),
		       gen.published_at
		FROM procedure_version_documents link
		JOIN documents d ON d.id = link.document_id
		LEFT JOIN document_index_generations gen
		  ON gen.id = link.active_generation_id
		 AND gen.document_id = link.document_id
		 AND gen.procedure_version_id = link.procedure_version_id
		 AND gen.xa_id = link.xa_id
		WHERE link.procedure_version_id = $1
		ORDER BY link.document_id
		FOR UPDATE OF link`, versionID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := make([]sourceRow, 0)
	for rows.Next() {
		var src sourceRow
		if err := rows.Scan(
			&src.DocumentID, &src.Title, &src.RelationshipType, &src.IndexStatus,
			&src.ValidityStatus, &src.Unlinked,
			&src.XaID, &src.DomainID, &src.ProcedureID, &src.ProcedureVersionID,
			&src.ActiveGenerationID,
			&src.GenerationStatus, &src.ChunkCount, &src.VectorCount, &src.VectorDimension,
			&src.EmbeddingModelID, &src.PublishedAt,
		); err != nil {
			return nil, err
		}
		out = append(out, src)
	}
	return out, rows.Err()
}

func writeAudit(ctx context.Context, tx pgx.Tx, requestID, actor uuid.UUID, action, entityType, entityID string, payload map[string]any) error {
	raw, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	_, err = tx.Exec(ctx, `
		INSERT INTO audit_logs (request_id, actor_user_id, action, entity_type, entity_id, payload)
		VALUES ($1,$2,$3,$4,$5,$6::jsonb)`, requestID, actor, action, entityType, entityID, raw)
	return err
}

func listProcedures(ctx context.Context, pool *pgxpool.Pool, xaID string) ([]ProcedureSummary, error) {
	rows, err := pool.Query(ctx, `
		SELECT p.id, p.procedure_code, p.name, p.domain_id, d.name, p.xa_id,
		       p.active_version_id, COALESCE(pv.version, ''),
		       (SELECT count(*) FROM procedure_versions v WHERE v.procedure_id = p.id),
		       COALESCE(pv.activated_at IS NULL AND pv.status = 'ACTIVE', false)
		FROM procedures p
		JOIN domains d ON d.id = p.domain_id
		LEFT JOIN procedure_versions pv ON pv.id = p.active_version_id AND pv.procedure_id = p.id
		WHERE p.xa_id = $1
		ORDER BY d.sort_order, p.procedure_code`, xaID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := make([]ProcedureSummary, 0)
	for rows.Next() {
		var item ProcedureSummary
		if err := rows.Scan(
			&item.ID, &item.ProcedureCode, &item.Name, &item.DomainID, &item.DomainName, &item.XaID,
			&item.ActiveVersionID, &item.ActiveVersion, &item.VersionCount, &item.LegacyActive,
		); err != nil {
			return nil, err
		}
		out = append(out, item)
	}
	return out, rows.Err()
}

func listVersions(ctx context.Context, pool *pgxpool.Pool, xaID string, procedureID uuid.UUID) ([]VersionSummary, error) {
	rows, err := pool.Query(ctx, `
		SELECT pv.id, pv.procedure_id, pv.version, pv.status, pv.created_at, pv.created_by,
		       pv.approved_by, pv.approved_at, pv.activated_at, pv.activated_by, pv.definition,
		       p.xa_id, p.domain_id, p.procedure_code, p.name
		FROM procedure_versions pv
		JOIN procedures p ON p.id = pv.procedure_id
		WHERE p.xa_id = $1 AND pv.procedure_id = $2
		ORDER BY pv.created_at, pv.version`, xaID, procedureID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	base := make([]versionRow, 0)
	ids := make([]uuid.UUID, 0)
	for rows.Next() {
		var row versionRow
		if err := rows.Scan(
			&row.ID, &row.ProcedureID, &row.Version, &row.Status, &row.CreatedAt, &row.CreatedBy,
			&row.ApprovedBy, &row.ApprovedAt, &row.ActivatedAt, &row.ActivatedBy, &row.Definition,
			&row.XaID, &row.DomainID, &row.ProcedureCode, &row.ProcedureName,
		); err != nil {
			return nil, err
		}
		base = append(base, row)
		ids = append(ids, row.ID)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	grouped := map[uuid.UUID][]sourceRow{}
	if len(ids) > 0 {
		srcRows, err := pool.Query(ctx, `
			SELECT link.procedure_version_id, link.document_id, d.title, link.relationship_type, link.index_status,
			       d.validity_status, link.unlinked_at IS NOT NULL,
			       link.xa_id, link.domain_id, link.procedure_id, link.procedure_version_id,
			       link.active_generation_id,
			       COALESCE(gen.status, ''),
			       COALESCE(gen.chunk_count, 0),
			       COALESCE(gen.vector_count, 0),
			       COALESCE(gen.vector_dimension, 0),
			       COALESCE(gen.embedding_model_id, ''),
			       gen.published_at
			FROM procedure_version_documents link
			JOIN documents d ON d.id = link.document_id
			LEFT JOIN document_index_generations gen
			  ON gen.id = link.active_generation_id
			 AND gen.document_id = link.document_id
			 AND gen.procedure_version_id = link.procedure_version_id
			 AND gen.xa_id = link.xa_id
			WHERE link.procedure_version_id = ANY($1)`, ids)
		if err != nil {
			return nil, err
		}
		defer srcRows.Close()
		for srcRows.Next() {
			var versionID uuid.UUID
			var src sourceRow
			if err := srcRows.Scan(
				&versionID, &src.DocumentID, &src.Title, &src.RelationshipType, &src.IndexStatus,
				&src.ValidityStatus, &src.Unlinked,
				&src.XaID, &src.DomainID, &src.ProcedureID, &src.ProcedureVersionID,
				&src.ActiveGenerationID,
				&src.GenerationStatus, &src.ChunkCount, &src.VectorCount, &src.VectorDimension,
				&src.EmbeddingModelID, &src.PublishedAt,
			); err != nil {
				return nil, err
			}
			grouped[versionID] = append(grouped[versionID], src)
		}
		if err := srcRows.Err(); err != nil {
			return nil, err
		}
	}
	out := make([]VersionSummary, 0, len(base))
	for _, row := range base {
		out = append(out, toSummary(row, grouped[row.ID]))
	}
	return out, nil
}

func toSummary(row versionRow, sources []sourceRow) VersionSummary {
	docs := make([]DocumentLink, 0, len(sources))
	live := make([]sourceRow, 0, len(sources))
	for _, src := range sources {
		if src.Unlinked {
			continue
		}
		docs = append(docs, documentView(src, row))
		live = append(live, src)
	}
	el := evaluateEligibility(row, live)
	return VersionSummary{
		ID: row.ID, ProcedureID: row.ProcedureID, Version: row.Version, Status: row.Status,
		CreatedAt: row.CreatedAt, CreatedBy: row.CreatedBy, ApprovedBy: row.ApprovedBy,
		ApprovedAt: row.ApprovedAt, ActivatedAt: row.ActivatedAt, ActivatedBy: row.ActivatedBy,
		Definition: row.Definition, Documents: docs, Eligibility: el,
		LegacyActive: row.Status == "ACTIVE" && row.ActivatedAt == nil,
	}
}

func listHistory(ctx context.Context, pool *pgxpool.Pool, procedureID uuid.UUID) ([]HistoryItem, error) {
	rows, err := pool.Query(ctx, `
		SELECT request_id, actor_user_id, created_at, payload
		FROM audit_logs
		WHERE action = $1 AND entity_type = 'procedure' AND entity_id = $2
		ORDER BY created_at DESC, id DESC
		LIMIT 50`, AuditAction, procedureID.String())
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := make([]HistoryItem, 0)
	for rows.Next() {
		var item HistoryItem
		var payload []byte
		if err := rows.Scan(&item.RequestID, &item.ActorUserID, &item.CreatedAt, &payload); err != nil {
			return nil, err
		}
		var meta map[string]any
		_ = json.Unmarshal(payload, &meta)
		item.ProcedureID = procedureID
		if v, ok := meta["new_active_version_id"].(string); ok {
			if id, err := uuid.Parse(v); err == nil {
				item.NewActiveVersionID = id
			}
		}
		if v, ok := meta["previous_active_version_id"].(string); ok && v != "" {
			if id, err := uuid.Parse(v); err == nil {
				item.PreviousActiveVersionID = &id
			}
		}
		out = append(out, item)
	}
	return out, rows.Err()
}

func isUniqueViolation(err error) bool {
	var pgErr *pgconn.PgError
	return errors.As(err, &pgErr) && pgErr.Code == "23505"
}

func nextVersionLabel(ctx context.Context, tx pgx.Tx, procedureID uuid.UUID) (string, error) {
	var labels []string
	rows, err := tx.Query(ctx, `SELECT version FROM procedure_versions WHERE procedure_id = $1 AND status <> 'SOURCE'`, procedureID)
	if err != nil {
		return "", err
	}
	defer rows.Close()
	for rows.Next() {
		var v string
		if err := rows.Scan(&v); err != nil {
			return "", err
		}
		labels = append(labels, v)
	}
	if err := rows.Err(); err != nil {
		return "", err
	}
	return bumpVersion(labels), nil
}

func bumpVersion(labels []string) string {
	major, minor, patch := 1, 0, 0
	found := false
	for _, label := range labels {
		var a, b, c int
		if _, err := fmt.Sscanf(label, "%d.%d.%d", &a, &b, &c); err != nil {
			continue
		}
		if !found || a > major || (a == major && b > minor) || (a == major && b == minor && c > patch) {
			major, minor, patch = a, b, c
			found = true
		}
	}
	if !found {
		return "1.0.0"
	}
	return fmt.Sprintf("%d.%d.%d", major, minor, patch+1)
}

func nowUTC() time.Time {
	return time.Now().UTC()
}
