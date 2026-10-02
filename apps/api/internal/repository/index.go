package repository

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"strings"
	"time"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/index"
	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
)

type IndexRepo struct {
	Pool *pgxpool.Pool
}

func (r *IndexRepo) ListTargets(ctx context.Context, xaID string) ([]index.Target, error) {
	rows, err := r.Pool.Query(ctx, `
		SELECT p.id, p.procedure_code, p.name, p.domain_id, v.id, v.version, v.status
		FROM procedures p
		JOIN procedure_versions v ON v.procedure_id = p.id
		WHERE p.xa_id = $1 AND v.status <> 'ARCHIVED'
		ORDER BY p.procedure_code, v.version`, xaID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := make([]index.Target, 0)
	for rows.Next() {
		var item index.Target
		if err := rows.Scan(&item.ProcedureID, &item.ProcedureCode, &item.ProcedureName, &item.DomainID, &item.ProcedureVersionID, &item.Version, &item.Status); err != nil {
			return nil, err
		}
		out = append(out, item)
	}
	return out, rows.Err()
}

func (r *IndexRepo) ListLinks(ctx context.Context, xaID string, documentID uuid.UUID) ([]index.Link, error) {
	var exists bool
	if err := r.Pool.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM documents WHERE id = $1 AND xa_id = $2)`, documentID, xaID).Scan(&exists); err != nil {
		return nil, err
	}
	if !exists {
		return nil, index.ErrNotFound
	}
	rows, err := r.Pool.Query(ctx, `
		SELECT link.document_id, link.procedure_id, link.procedure_version_id,
			p.procedure_code, v.version, link.relationship_type, link.page_range,
			link.index_status, link.last_error_code, link.reindex_error_code, link.updated_at, job.claim_expires_at,
			gen.page_count, gen.native_page_count, gen.ocr_page_count, gen.chunk_count,
			gen.pipeline_version, gen.embedding_model_id
		FROM procedure_version_documents link
		JOIN procedures p ON p.id = link.procedure_id
		JOIN procedure_versions v ON v.id = link.procedure_version_id AND v.procedure_id = link.procedure_id
		LEFT JOIN LATERAL (
			SELECT claim_expires_at
			FROM document_index_jobs
			WHERE document_id = link.document_id
			  AND procedure_version_id = link.procedure_version_id
			  AND xa_id = link.xa_id
			  AND status = 'CLAIMED'
			ORDER BY claimed_at DESC
			LIMIT 1
		) job ON true
		LEFT JOIN document_index_generations gen
			ON gen.id = link.active_generation_id
			AND gen.document_id = link.document_id
			AND gen.procedure_version_id = link.procedure_version_id
			AND gen.xa_id = link.xa_id
			AND gen.status = 'READY'
		WHERE link.document_id = $1 AND link.xa_id = $2 AND link.unlinked_at IS NULL
		ORDER BY link.created_at`, documentID, xaID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	now := time.Now()
	out := make([]index.Link, 0)
	for rows.Next() {
		var item index.Link
		var pageCount, nativeCount, ocrCount, chunkCount *int
		var pipelineVersion, modelID *string
		if err := rows.Scan(&item.DocumentID, &item.ProcedureID, &item.ProcedureVersionID, &item.ProcedureCode, &item.Version, &item.RelationshipType, &item.PageRange, &item.IndexStatus, &item.LastErrorCode, &item.ReindexErrorCode, &item.UpdatedAt, &item.ClaimExpiresAt, &pageCount, &nativeCount, &ocrCount, &chunkCount, &pipelineVersion, &modelID); err != nil {
			return nil, err
		}
		item.PageCount = intPtr(pageCount)
		item.NativePageCount = intPtr(nativeCount)
		item.OCRPageCount = intPtr(ocrCount)
		item.ChunkCount = intPtr(chunkCount)
		item.PipelineVersion = pipelineVersion
		item.EmbeddingModelID = modelID
		index.MarkRecoverable(&item, now)
		out = append(out, item)
	}
	return out, rows.Err()
}

func (r *IndexRepo) Link(ctx context.Context, xaID string, in index.LinkInput, payloadHash string) (index.Link, bool, error) {
	tx, err := r.Pool.Begin(ctx)
	if err != nil {
		return index.Link{}, false, err
	}
	defer tx.Rollback(ctx)
	var docDomain *string
	err = tx.QueryRow(ctx, `SELECT domain_id FROM documents WHERE id = $1 AND xa_id = $2 FOR UPDATE`, in.DocumentID, xaID).Scan(&docDomain)
	if errors.Is(err, pgx.ErrNoRows) {
		return index.Link{}, false, index.ErrNotFound
	}
	if err != nil {
		return index.Link{}, false, err
	}
	if replay, ok, err := claimIdempotency(ctx, tx, xaID, in.RequestID, "LINK", payloadHash); err != nil {
		return index.Link{}, false, err
	} else if ok {
		var link index.Link
		if err := json.Unmarshal(replay, &link); err != nil {
			return index.Link{}, false, err
		}
		if link.DocumentID == uuid.Nil {
			link.DocumentID = in.DocumentID
		}
		if link.ProcedureVersionID == uuid.Nil {
			link.ProcedureVersionID = in.ProcedureVersionID
		}
		if link.DocumentID != in.DocumentID || link.ProcedureVersionID != in.ProcedureVersionID {
			return index.Link{}, false, index.ErrIdempotency
		}
		if err := fillLegacyLink(ctx, tx, xaID, in.RequestID, &link, replay); err != nil {
			return index.Link{}, false, err
		}
		return link, true, nil
	}
	var procedureID uuid.UUID
	var domainID, code, version string
	err = tx.QueryRow(ctx, `
		SELECT p.id, p.domain_id, p.procedure_code, v.version
		FROM procedure_versions v
		JOIN procedures p ON p.id = v.procedure_id
		WHERE v.id = $1 AND p.xa_id = $2 AND v.status <> 'ARCHIVED'`,
		in.ProcedureVersionID, xaID).Scan(&procedureID, &domainID, &code, &version)
	if errors.Is(err, pgx.ErrNoRows) {
		return index.Link{}, false, index.ErrNotFound
	}
	if err != nil {
		return index.Link{}, false, err
	}
	if docDomain == nil || *docDomain != domainID {
		return index.Link{}, false, index.ErrValidation
	}
	var page any
	if in.PageRange != "" {
		page = in.PageRange
	}
	var updated time.Time
	if _, err = tx.Exec(ctx, `SAVEPOINT relink`); err != nil {
		return index.Link{}, false, err
	}
	err = tx.QueryRow(ctx, `
		INSERT INTO procedure_version_documents (
			procedure_version_id, document_id, procedure_id, xa_id, domain_id,
			relationship_type, page_range, index_status
		) VALUES ($1,$2,$3,$4,$5,$6,$7,'UPLOADED')
		RETURNING updated_at`,
		in.ProcedureVersionID, in.DocumentID, procedureID, xaID, domainID, in.RelationshipType, page).Scan(&updated)
	if err != nil {
		var pgErr *pgconn.PgError
		if errors.As(err, &pgErr) && pgErr.Code == "23505" {
			if _, err = tx.Exec(ctx, `ROLLBACK TO SAVEPOINT relink`); err != nil {
				return index.Link{}, false, err
			}
			err = tx.QueryRow(ctx, `
				UPDATE procedure_version_documents
				SET unlinked_at = NULL, index_status = 'UPLOADED', active_generation_id = NULL,
					last_error_code = NULL, reindex_error_code = NULL,
					relationship_type = $4, page_range = $5, updated_at = now()
				WHERE procedure_version_id = $1 AND document_id = $2 AND xa_id = $3
				  AND unlinked_at IS NOT NULL
				RETURNING updated_at`,
				in.ProcedureVersionID, in.DocumentID, xaID, in.RelationshipType, page).Scan(&updated)
			if errors.Is(err, pgx.ErrNoRows) {
				return index.Link{}, false, index.ErrDuplicate
			}
			if err != nil {
				return index.Link{}, false, err
			}
		} else if errors.As(err, &pgErr) && pgErr.Code == "23503" {
			return index.Link{}, false, index.ErrValidation
		} else {
			return index.Link{}, false, err
		}
	}
	if _, err = tx.Exec(ctx, `RELEASE SAVEPOINT relink`); err != nil {
		return index.Link{}, false, err
	}
	if err := recomputeDocumentStatus(ctx, tx, xaID, in.DocumentID); err != nil {
		return index.Link{}, false, err
	}
	link := index.Link{
		DocumentID: in.DocumentID, ProcedureID: procedureID, ProcedureVersionID: in.ProcedureVersionID,
		ProcedureCode: code, Version: version, RelationshipType: in.RelationshipType,
		IndexStatus: "UPLOADED", UpdatedAt: updated,
	}
	if in.PageRange != "" {
		link.PageRange = &in.PageRange
	}
	raw, _ := json.Marshal(link)
	if err := writeAudit(ctx, tx, in.RequestID, in.ActorUserID, "DOCUMENT_LINKED", in.DocumentID, map[string]any{
		"procedure_version_id": in.ProcedureVersionID.String(),
		"relationship_type":    in.RelationshipType,
		"xa_id":                xaID,
	}); err != nil {
		return index.Link{}, false, err
	}
	if err := storeIdempotency(ctx, tx, xaID, in.RequestID, "LINK", payloadHash, 201, raw); err != nil {
		return index.Link{}, false, err
	}
	if err := tx.Commit(ctx); err != nil {
		return index.Link{}, false, err
	}
	return link, false, nil
}

func (r *IndexRepo) Unlink(ctx context.Context, xaID string, documentID, versionID, actor, requestID uuid.UUID, payloadHash string) (bool, error) {
	tx, err := r.Pool.Begin(ctx)
	if err != nil {
		return false, err
	}
	defer tx.Rollback(ctx)
	var exists bool
	if err := tx.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM documents WHERE id = $1 AND xa_id = $2)`, documentID, xaID).Scan(&exists); err != nil {
		return false, err
	}
	if !exists {
		return false, index.ErrNotFound
	}
	if _, err := tx.Exec(ctx, `SELECT id FROM documents WHERE id = $1 AND xa_id = $2 FOR UPDATE`, documentID, xaID); err != nil {
		return false, err
	}
	if replay, ok, err := claimIdempotency(ctx, tx, xaID, requestID, "UNLINK", payloadHash); err != nil {
		return false, err
	} else if ok {
		var stored struct {
			DocumentID         uuid.UUID `json:"document_id"`
			ProcedureVersionID uuid.UUID `json:"procedure_version_id"`
		}
		if err := json.Unmarshal(replay, &stored); err != nil {
			return false, err
		}
		// v10 stored only {"status":"unlinked"}. The payload hash already binds
		// the document and version, so missing ids are still a replay.
		if (stored.DocumentID != uuid.Nil && stored.DocumentID != documentID) ||
			(stored.ProcedureVersionID != uuid.Nil && stored.ProcedureVersionID != versionID) {
			return false, index.ErrIdempotency
		}
		return true, nil
	}
	var linkStatus string
	err = tx.QueryRow(ctx, `
		SELECT index_status FROM procedure_version_documents
		WHERE document_id = $1 AND procedure_version_id = $2 AND xa_id = $3
		FOR UPDATE`, documentID, versionID, xaID).Scan(&linkStatus)
	if errors.Is(err, pgx.ErrNoRows) {
		return false, index.ErrNotFound
	}
	if err != nil {
		return false, err
	}
	if linkStatus == "PROCESSING" {
		return false, index.ErrConflict
	}
	var live bool
	if err := tx.QueryRow(ctx, `
		SELECT EXISTS(
			SELECT 1 FROM document_index_jobs
			WHERE document_id = $1 AND procedure_version_id = $2 AND xa_id = $3
			  AND status = 'CLAIMED' AND claim_expires_at > now()
			FOR UPDATE
		)`, documentID, versionID, xaID).Scan(&live); err != nil {
		return false, err
	}
	if live {
		return false, index.ErrConflict
	}
	if _, err := tx.Exec(ctx, `
		UPDATE document_index_generations
		SET status = 'SUPERSEDED', updated_at = now()
		WHERE document_id = $1 AND procedure_version_id = $2 AND xa_id = $3 AND status = 'READY'`,
		documentID, versionID, xaID); err != nil {
		return false, err
	}
	tag, err := tx.Exec(ctx, `
		UPDATE procedure_version_documents
		SET unlinked_at = now(), active_generation_id = NULL, updated_at = now()
		WHERE document_id = $1 AND procedure_version_id = $2 AND xa_id = $3 AND unlinked_at IS NULL`,
		documentID, versionID, xaID)
	if err != nil {
		return false, err
	}
	if tag.RowsAffected() == 0 {
		return false, index.ErrNotFound
	}
	if err := recomputeDocumentStatus(ctx, tx, xaID, documentID); err != nil {
		return false, err
	}
	body, _ := json.Marshal(map[string]string{
		"status":               "unlinked",
		"document_id":          documentID.String(),
		"procedure_version_id": versionID.String(),
	})
	if err := writeAudit(ctx, tx, requestID, actor, "DOCUMENT_UNLINKED", documentID, map[string]any{
		"procedure_version_id": versionID.String(),
		"xa_id":                xaID,
	}); err != nil {
		return false, err
	}
	if err := storeIdempotency(ctx, tx, xaID, requestID, "UNLINK", payloadHash, 200, body); err != nil {
		return false, err
	}
	return false, tx.Commit(ctx)
}

func (r *IndexRepo) Claim(ctx context.Context, in index.ClaimInput) (index.ClaimResult, error) {
	tx, err := r.Pool.Begin(ctx)
	if err != nil {
		return index.ClaimResult{}, err
	}
	defer tx.Rollback(ctx)
	var docDomain *string
	err = tx.QueryRow(ctx, `SELECT domain_id FROM documents WHERE id = $1 AND xa_id = $2 FOR UPDATE`, in.DocumentID, in.XAID).Scan(&docDomain)
	if errors.Is(err, pgx.ErrNoRows) {
		return index.ClaimResult{}, index.ErrNotFound
	}
	if err != nil {
		return index.ClaimResult{}, err
	}
	var linkStatus, relationship, checksum, domain string
	var page *string
	var unlinkedAt *time.Time
	err = tx.QueryRow(ctx, `
		SELECT link.index_status, link.relationship_type, link.page_range, d.checksum, d.domain_id, link.unlinked_at
		FROM procedure_version_documents link
		JOIN documents d ON d.id = link.document_id AND d.xa_id = link.xa_id
		WHERE link.document_id = $1 AND link.procedure_version_id = $2 AND link.xa_id = $3
		FOR UPDATE OF link`, in.DocumentID, in.ProcedureVersionID, in.XAID).Scan(&linkStatus, &relationship, &page, &checksum, &domain, &unlinkedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return index.ClaimResult{}, index.ErrNotLinked
	}
	if err != nil {
		return index.ClaimResult{}, err
	}
	if unlinkedAt != nil {
		return index.ClaimResult{}, index.ErrNotLinked
	}
	if docDomain == nil || domain == "" {
		return index.ClaimResult{}, index.ErrValidation
	}
	workerReq := index.Request{
		SchemaVersion: index.SchemaVersion, DocumentID: in.DocumentID, XAID: in.XAID,
		DomainID: domain, Checksum: checksum, ProcedureVersionID: in.ProcedureVersionID,
		RelationshipType: relationship, PageRange: page,
	}
	var jobID, token, jobDoc, jobVer uuid.UUID
	var status, hash, errCode string
	var expires *time.Time
	err = tx.QueryRow(ctx, `
		SELECT id, claim_token, status, payload_hash, coalesce(error_code, ''), claim_expires_at,
			document_id, procedure_version_id
		FROM document_index_jobs
		WHERE xa_id = $1 AND request_id = $2
		FOR UPDATE`, in.XAID, in.RequestID).Scan(&jobID, &token, &status, &hash, &errCode, &expires, &jobDoc, &jobVer)
	if err == nil {
		if hash != in.PayloadHash || jobDoc != in.DocumentID || jobVer != in.ProcedureVersionID {
			return index.ClaimResult{}, index.ErrIdempotency
		}
		docStatus, err := documentStatus(ctx, tx, in.XAID, in.DocumentID)
		if err != nil {
			return index.ClaimResult{}, err
		}
		switch status {
		case "SUCCEEDED":
			return index.ClaimResult{JobID: jobID, ClaimToken: token, LinkStatus: "READY", DocumentStatus: docStatus}, tx.Commit(ctx)
		case "FAILED":
			return index.ClaimResult{JobID: jobID, ClaimToken: token, LinkStatus: "FAILED", DocumentStatus: docStatus, ErrorCode: errCode}, tx.Commit(ctx)
		case "CLAIMED":
			if expires != nil && expires.After(time.Now()) {
				return index.ClaimResult{JobID: jobID, ClaimToken: token, LinkStatus: "PROCESSING", DocumentStatus: docStatus}, tx.Commit(ctx)
			}
			token = uuid.New()
			lease := time.Now().Add(in.TTL)
			if _, err := tx.Exec(ctx, `
				UPDATE document_index_jobs
				SET claim_token = $2, claimed_at = now(), claim_expires_at = $3
				WHERE id = $1 AND xa_id = $4 AND status = 'CLAIMED'`, jobID, token, lease, in.XAID); err != nil {
				return index.ClaimResult{}, err
			}
			if err := tx.Commit(ctx); err != nil {
				return index.ClaimResult{}, err
			}
			return index.ClaimResult{JobID: jobID, ClaimToken: token, RunWorker: true, WorkerRequest: workerReq}, nil
		}
	}
	if err != nil && !errors.Is(err, pgx.ErrNoRows) {
		return index.ClaimResult{}, err
	}
	if _, err := tx.Exec(ctx, `
		UPDATE document_index_jobs
		SET status = 'FAILED', error_code = 'lease_expired', finished_at = now(), claim_expires_at = NULL
		WHERE document_id = $1 AND procedure_version_id = $2 AND xa_id = $3
		  AND status = 'CLAIMED' AND claim_expires_at <= now()`, in.DocumentID, in.ProcedureVersionID, in.XAID); err != nil {
		return index.ClaimResult{}, err
	}
	var live bool
	if err := tx.QueryRow(ctx, `
		SELECT EXISTS(
			SELECT 1 FROM document_index_jobs
			WHERE document_id = $1 AND procedure_version_id = $2 AND xa_id = $3
			  AND status = 'CLAIMED' AND claim_expires_at > now()
		)`, in.DocumentID, in.ProcedureVersionID, in.XAID).Scan(&live); err != nil {
		return index.ClaimResult{}, err
	}
	if live {
		return index.ClaimResult{}, index.ErrConflict
	}
	switch {
	case in.Reindex && linkStatus == "READY":
	case !in.Retry && !in.Reindex && linkStatus == "UPLOADED":
	case in.Retry && linkStatus == "FAILED":
	case in.Retry && linkStatus == "PROCESSING":
	default:
		return index.ClaimResult{}, index.ErrInvalidState
	}
	jobID = uuid.New()
	token = uuid.New()
	lease := time.Now().Add(in.TTL)
	_, err = tx.Exec(ctx, `
		INSERT INTO document_index_jobs (
			id, document_id, procedure_version_id, xa_id, request_id, payload_hash,
			status, claim_token, claimed_at, claim_expires_at
		) VALUES ($1,$2,$3,$4,$5,$6,'CLAIMED',$7,$8,$9)`,
		jobID, in.DocumentID, in.ProcedureVersionID, in.XAID, in.RequestID, in.PayloadHash, token, time.Now(), lease)
	if err != nil {
		var pgErr *pgconn.PgError
		if errors.As(err, &pgErr) && pgErr.Code == "23505" {
			return index.ClaimResult{}, index.ErrConflict
		}
		return index.ClaimResult{}, err
	}
	if !in.Reindex {
		if _, err := tx.Exec(ctx, `
			UPDATE procedure_version_documents
			SET index_status = 'PROCESSING', last_error_code = NULL, updated_at = now()
			WHERE document_id = $1 AND procedure_version_id = $2 AND xa_id = $3`,
			in.DocumentID, in.ProcedureVersionID, in.XAID); err != nil {
			return index.ClaimResult{}, err
		}
	}
	if err := recomputeDocumentStatus(ctx, tx, in.XAID, in.DocumentID); err != nil {
		return index.ClaimResult{}, err
	}
	if err := writeAudit(ctx, tx, in.RequestID, in.ActorUserID, in.AuditAction, in.DocumentID, map[string]any{
		"xa_id": in.XAID, "job_id": jobID.String(), "procedure_version_id": in.ProcedureVersionID.String(),
	}); err != nil {
		return index.ClaimResult{}, err
	}
	if err := tx.Commit(ctx); err != nil {
		return index.ClaimResult{}, err
	}
	return index.ClaimResult{JobID: jobID, ClaimToken: token, RunWorker: true, WorkerRequest: workerReq}, nil
}

func intPtr(v *int) *int {
	if v == nil {
		return nil
	}
	copied := *v
	return &copied
}

func (r *IndexRepo) StageGeneration(ctx context.Context, in index.StageInput) (index.StagedObject, error) {
	var procedureID uuid.UUID
	var uri, checksum string
	err := r.Pool.QueryRow(ctx, `
		SELECT link.procedure_id, d.storage_uri, d.checksum
		FROM procedure_version_documents link
		JOIN documents d ON d.id = link.document_id AND d.xa_id = link.xa_id
		WHERE link.document_id = $1 AND link.procedure_version_id = $2 AND link.xa_id = $3`,
		in.DocumentID, in.ProcedureVersionID, in.XAID).Scan(&procedureID, &uri, &checksum)
	if err != nil {
		return index.StagedObject{}, err
	}
	bucket, key, ok := splitInternalURI(uri)
	if !ok || bucket != in.Bucket {
		return index.StagedObject{}, index.ErrValidation
	}
	id := uuid.New()
	_, err = r.Pool.Exec(ctx, `
		INSERT INTO document_index_generations (
			id, xa_id, document_id, procedure_version_id, job_id, status,
			pipeline_version, extraction_version, ocr_version, chunk_config_hash,
			embedding_model_id, embedding_revision, embedding_checksum, vector_dimension,
			source_sha256, content_sha256, manifest_hash
		) VALUES (
			$1,$2,$3,$4,$5,'STAGING',
			'pending','pending','pending','pending',
			'pending','pending','',0,
			$6,'',''
		)`, id, in.XAID, in.DocumentID, in.ProcedureVersionID, in.JobID, checksum)
	if err != nil {
		return index.StagedObject{}, err
	}
	return index.StagedObject{GenerationID: id, ProcedureID: procedureID, Bucket: bucket, ObjectKey: key, Checksum: checksum}, nil
}

func splitInternalURI(uri string) (string, string, bool) {
	const prefix = "s3://"
	if !strings.HasPrefix(uri, prefix) {
		return "", "", false
	}
	rest := strings.TrimPrefix(uri, prefix)
	bucket, key, ok := strings.Cut(rest, "/")
	if !ok || bucket == "" || key == "" || strings.Contains(key, "://") {
		return "", "", false
	}
	return bucket, key, true
}

func (r *IndexRepo) Finish(ctx context.Context, in index.FinishInput) (index.FinishResult, error) {
	tx, err := r.Pool.Begin(ctx)
	if err != nil {
		return index.FinishResult{}, err
	}
	defer tx.Rollback(ctx)
	if _, err := tx.Exec(ctx, `SELECT id FROM documents WHERE id = $1 AND xa_id = $2 FOR UPDATE`, in.DocumentID, in.XAID); err != nil {
		return index.FinishResult{}, err
	}
	var unlinkedAt *time.Time
	err = tx.QueryRow(ctx, `
		SELECT unlinked_at FROM procedure_version_documents
		WHERE document_id = $1 AND procedure_version_id = $2 AND xa_id = $3
		FOR UPDATE`, in.DocumentID, in.ProcedureVersionID, in.XAID).Scan(&unlinkedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return index.FinishResult{}, tx.Commit(ctx)
	}
	if err != nil {
		return index.FinishResult{}, err
	}
	if unlinkedAt != nil {
		tag, err := tx.Exec(ctx, `
			UPDATE document_index_jobs
			SET status = 'FAILED', error_code = 'claim_rejected', finished_at = now(), claim_expires_at = NULL
			WHERE id = $1 AND claim_token = $2 AND xa_id = $3 AND status = 'CLAIMED'
			  AND document_id = $4 AND procedure_version_id = $5`,
			in.JobID, in.ClaimToken, in.XAID, in.DocumentID, in.ProcedureVersionID)
		if err != nil {
			return index.FinishResult{}, err
		}
		if tag.RowsAffected() == 1 && in.GenerationID != uuid.Nil {
			if _, err := tx.Exec(ctx, `
				UPDATE document_index_generations
				SET status = 'FAILED', error_code = 'claim_rejected', updated_at = now()
				WHERE id = $1 AND xa_id = $2 AND document_id = $3 AND procedure_version_id = $4 AND status = 'STAGING'`,
				in.GenerationID, in.XAID, in.DocumentID, in.ProcedureVersionID); err != nil {
				return index.FinishResult{}, err
			}
		}
		return index.FinishResult{Applied: false}, tx.Commit(ctx)
	}
	var tokenOK bool
	err = tx.QueryRow(ctx, `
		SELECT true FROM document_index_jobs
		WHERE id = $1 AND claim_token = $2 AND xa_id = $3 AND status = 'CLAIMED'
		  AND document_id = $4 AND procedure_version_id = $5
		FOR UPDATE`, in.JobID, in.ClaimToken, in.XAID, in.DocumentID, in.ProcedureVersionID).Scan(&tokenOK)
	if errors.Is(err, pgx.ErrNoRows) {
		return index.FinishResult{}, tx.Commit(ctx)
	}
	if err != nil {
		return index.FinishResult{}, err
	}
	if in.GenerationID != uuid.Nil && in.Outcome == index.OutcomeReady {
		ready, err := generationMatches(ctx, tx, in)
		if err != nil {
			return index.FinishResult{}, err
		}
		if !ready {
			in.Outcome = index.OutcomeFailed
			in.ErrorCode = "publish_mismatch"
		}
	}
	status := "FAILED"
	linkStatus := "FAILED"
	action := "DOCUMENT_INDEX_FAILED"
	keepReady := false
	if in.Outcome == index.OutcomeReady {
		status = "SUCCEEDED"
		linkStatus = "READY"
		action = "DOCUMENT_INDEX_READY"
	} else {
		var active *uuid.UUID
		if err := tx.QueryRow(ctx, `
			SELECT active_generation_id FROM procedure_version_documents
			WHERE document_id = $1 AND procedure_version_id = $2 AND xa_id = $3`,
			in.DocumentID, in.ProcedureVersionID, in.XAID).Scan(&active); err != nil {
			return index.FinishResult{}, err
		}
		if active != nil {
			linkStatus = "READY"
			keepReady = true
			action = "DOCUMENT_REINDEX_FAILED"
		}
	}
	code := in.ErrorCode
	if len(code) > index.MaxErrorCodeLen {
		code = "worker_failed"
	}
	tag, err := tx.Exec(ctx, `
		UPDATE document_index_jobs
		SET status = $2, error_code = NULLIF($3, ''), finished_at = now(), claim_expires_at = NULL
		WHERE id = $1 AND claim_token = $4 AND xa_id = $5 AND status = 'CLAIMED'
		  AND claim_expires_at > now()
		  AND document_id = $6 AND procedure_version_id = $7`,
		in.JobID, status, code, in.ClaimToken, in.XAID, in.DocumentID, in.ProcedureVersionID)
	if err != nil {
		return index.FinishResult{}, err
	}
	if tag.RowsAffected() == 0 {
		return index.FinishResult{}, tx.Commit(ctx)
	}
	if _, err := tx.Exec(ctx, `
		UPDATE procedure_version_documents
		SET index_status = $4,
			last_error_code = CASE WHEN $6 THEN last_error_code ELSE NULLIF($5, '') END,
			reindex_error_code = CASE WHEN $6 THEN NULLIF($5, '') WHEN $4 = 'READY' THEN NULL ELSE reindex_error_code END,
			updated_at = now()
		WHERE document_id = $1 AND procedure_version_id = $2 AND xa_id = $3`,
		in.DocumentID, in.ProcedureVersionID, in.XAID, linkStatus, code, keepReady); err != nil {
		return index.FinishResult{}, err
	}
	if in.GenerationID != uuid.Nil {
		if err := settleGeneration(ctx, tx, in, in.Outcome == index.OutcomeReady); err != nil {
			return index.FinishResult{}, err
		}
	}
	if err := recomputeDocumentStatus(ctx, tx, in.XAID, in.DocumentID); err != nil {
		return index.FinishResult{}, err
	}
	docStatus, err := documentStatus(ctx, tx, in.XAID, in.DocumentID)
	if err != nil {
		return index.FinishResult{}, err
	}
	if err := writeAudit(ctx, tx, in.RequestID, in.ActorUserID, action, in.DocumentID, map[string]any{
		"xa_id": in.XAID, "error_code": code, "procedure_version_id": in.ProcedureVersionID.String(),
	}); err != nil {
		return index.FinishResult{}, err
	}
	if err := tx.Commit(ctx); err != nil {
		return index.FinishResult{}, err
	}
	return index.FinishResult{Applied: true, LinkStatus: linkStatus, DocumentStatus: docStatus}, nil
}

func generationMatches(ctx context.Context, tx pgx.Tx, in index.FinishInput) (bool, error) {
	var status, manifest, source string
	var chunks, vectors int
	err := tx.QueryRow(ctx, `
		SELECT status, manifest_hash, source_sha256, chunk_count, vector_count
		FROM document_index_generations
		WHERE id = $1 AND xa_id = $2 AND document_id = $3 AND procedure_version_id = $4 AND job_id = $5
		FOR UPDATE`, in.GenerationID, in.XAID, in.DocumentID, in.ProcedureVersionID, in.JobID).
		Scan(&status, &manifest, &source, &chunks, &vectors)
	if err != nil {
		return false, err
	}
	if status != "STAGING" || manifest == "" || manifest != in.ManifestHash || chunks < 1 || chunks != in.ChunkCount || vectors != in.VectorCount || chunks != vectors {
		return false, nil
	}
	var checksum string
	if err := tx.QueryRow(ctx, `SELECT checksum FROM documents WHERE id = $1 AND xa_id = $2`, in.DocumentID, in.XAID).Scan(&checksum); err != nil {
		return false, err
	}
	if checksum != source {
		return false, nil
	}
	var count int
	var line string
	if err := tx.QueryRow(ctx, `
		SELECT count(*), coalesce(string_agg(
		chunk_index::text || '|' || page_start::text || '|' || page_end::text || '|' || text_sha256 || '|' || token_count::text || '|' || extraction_source,
			E'\n' ORDER BY chunk_index), '')
		FROM knowledge_chunks
		WHERE generation_id = $1 AND xa_id = $2`, in.GenerationID, in.XAID).Scan(&count, &line); err != nil {
		return false, err
	}
	if count != chunks {
		return false, nil
	}
	sum := sha256.Sum256([]byte(line))
	return hex.EncodeToString(sum[:]) == manifest, nil
}

func settleGeneration(ctx context.Context, tx pgx.Tx, in index.FinishInput, ready bool) error {
	if ready {
		if _, err := tx.Exec(ctx, `
			UPDATE document_index_generations
			SET status = 'SUPERSEDED', updated_at = now()
			WHERE document_id = $1 AND procedure_version_id = $2 AND xa_id = $3
			  AND status = 'READY' AND id <> $4`,
			in.DocumentID, in.ProcedureVersionID, in.XAID, in.GenerationID); err != nil {
			return err
		}
		tag, err := tx.Exec(ctx, `
			UPDATE document_index_generations
			SET status = 'READY', error_code = NULL, published_at = now(), updated_at = now()
			WHERE id = $1 AND xa_id = $2 AND document_id = $3 AND procedure_version_id = $4
			  AND job_id = $5 AND status = 'STAGING' AND cleanup_status = 'PENDING'`,
			in.GenerationID, in.XAID, in.DocumentID, in.ProcedureVersionID, in.JobID)
		if err != nil {
			return err
		}
		if tag.RowsAffected() != 1 {
			return index.ErrConflict
		}
		_, err = tx.Exec(ctx, `
			UPDATE procedure_version_documents
			SET active_generation_id = $4
			WHERE document_id = $1 AND procedure_version_id = $2 AND xa_id = $3`,
			in.DocumentID, in.ProcedureVersionID, in.XAID, in.GenerationID)
		return err
	}
	_, err := tx.Exec(ctx, `
		UPDATE document_index_generations
		SET status = 'FAILED', error_code = NULLIF($6, ''), updated_at = now()
		WHERE id = $1 AND xa_id = $2 AND document_id = $3 AND procedure_version_id = $4
		  AND job_id = $5 AND status = 'STAGING'`,
		in.GenerationID, in.XAID, in.DocumentID, in.ProcedureVersionID, in.JobID, in.ErrorCode)
	return err
}

func recomputeDocumentStatus(ctx context.Context, tx pgx.Tx, xaID string, documentID uuid.UUID) error {
	_, err := tx.Exec(ctx, `
		UPDATE documents AS doc
		SET processing_status = CASE
				WHEN NOT EXISTS (
					SELECT 1 FROM procedure_version_documents link
					WHERE link.document_id = doc.id AND link.xa_id = doc.xa_id AND link.unlinked_at IS NULL
				) THEN 'UPLOADED'
				WHEN NOT EXISTS (
					SELECT 1 FROM procedure_version_documents link
					WHERE link.document_id = doc.id AND link.xa_id = doc.xa_id AND link.unlinked_at IS NULL AND link.index_status <> 'READY'
				) THEN 'READY'
				WHEN EXISTS (
					SELECT 1 FROM procedure_version_documents link
					WHERE link.document_id = doc.id AND link.xa_id = doc.xa_id AND link.unlinked_at IS NULL AND link.index_status = 'PROCESSING'
				) THEN 'PROCESSING'
				WHEN EXISTS (
					SELECT 1 FROM procedure_version_documents link
					WHERE link.document_id = doc.id AND link.xa_id = doc.xa_id AND link.unlinked_at IS NULL AND link.index_status = 'FAILED'
				) THEN 'FAILED'
				ELSE 'UPLOADED'
			END,
			updated_at = now()
		WHERE doc.id = $1 AND doc.xa_id = $2`, documentID, xaID)
	return err
}

func (r *IndexRepo) GenerationMetrics(ctx context.Context, xaID string) (index.GenerationMetrics, error) {
	var out index.GenerationMetrics
	err := r.Pool.QueryRow(ctx, `
		SELECT
			count(*) FILTER (WHERE status = 'READY'),
			count(*) FILTER (WHERE status = 'STAGING'),
			count(*) FILTER (WHERE status = 'FAILED'),
			count(*) FILTER (
				WHERE status = 'STAGING' AND NOT EXISTS (
					SELECT 1 FROM document_index_jobs job
					WHERE job.id = document_index_generations.job_id
					  AND job.status = 'CLAIMED' AND job.claim_expires_at > now()
				)
			)
		FROM document_index_generations
		WHERE xa_id = $1`, xaID).Scan(&out.Ready, &out.Staging, &out.Failed, &out.Orphan)
	if err != nil {
		return out, err
	}
	err = r.Pool.QueryRow(ctx, `
		SELECT
			count(*) FILTER (WHERE action = 'DOCUMENT_INDEX_RETRY'),
			count(*) FILTER (WHERE action = 'DOCUMENT_REINDEX_REQUESTED')
		FROM audit_logs
		WHERE payload->>'xa_id' = $1`, xaID).Scan(&out.RetryCount, &out.ReindexCount)
	return out, err
}

func documentStatus(ctx context.Context, tx pgx.Tx, xaID string, documentID uuid.UUID) (string, error) {
	var status string
	err := tx.QueryRow(ctx, `SELECT processing_status FROM documents WHERE id = $1 AND xa_id = $2`, documentID, xaID).Scan(&status)
	return status, err
}

// claimIdempotency inserts the idempotency key before any later read, so two
// transactions with the same (xa_id, request_id, action) cannot both miss the
// row. The insert holds the row lock until commit; the waiter then replays.
func fillLegacyLink(ctx context.Context, tx pgx.Tx, xaID string, requestID uuid.UUID, link *index.Link, raw []byte) error {
	var probe map[string]json.RawMessage
	if err := json.Unmarshal(raw, &probe); err != nil {
		return err
	}
	_, hasRecoverable := probe["recoverable"]
	if link.IndexStatus != "" && !link.UpdatedAt.IsZero() && hasRecoverable {
		return nil
	}
	var status string
	var updated time.Time
	var expires *time.Time
	err := tx.QueryRow(ctx, `
		SELECT link.index_status, link.updated_at, job.claim_expires_at
		FROM procedure_version_documents AS link
		LEFT JOIN LATERAL (
			SELECT claim_expires_at
			FROM document_index_jobs
			WHERE document_id = link.document_id
			  AND procedure_version_id = link.procedure_version_id
			  AND xa_id = link.xa_id
			  AND status = 'CLAIMED'
			ORDER BY claimed_at DESC
			LIMIT 1
		) AS job ON true
		WHERE link.document_id = $1 AND link.procedure_version_id = $2 AND link.xa_id = $3`,
		link.DocumentID, link.ProcedureVersionID, xaID).Scan(&status, &updated, &expires)
	if err != nil && !errors.Is(err, pgx.ErrNoRows) {
		return err
	}
	if errors.Is(err, pgx.ErrNoRows) {
		if link.IndexStatus == "" {
			link.IndexStatus = "UPLOADED"
		}
		if link.UpdatedAt.IsZero() {
			var created time.Time
			if scanErr := tx.QueryRow(ctx, `
				SELECT created_at FROM admin_write_idempotency
				WHERE xa_id = $1 AND request_id = $2 AND action = 'LINK'`, xaID, requestID).Scan(&created); scanErr != nil {
				return scanErr
			}
			link.UpdatedAt = created
		}
	} else {
		if link.IndexStatus == "" {
			link.IndexStatus = status
		}
		if link.UpdatedAt.IsZero() {
			link.UpdatedAt = updated
		}
		if !hasRecoverable {
			link.ClaimExpiresAt = expires
		}
	}
	if link.IndexStatus == "" || link.UpdatedAt.IsZero() {
		return errors.New("legacy link replay is missing status")
	}
	if !hasRecoverable {
		index.MarkRecoverable(link, time.Now())
	}
	return nil
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
	if errors.Is(err, pgx.ErrNoRows) {
		tag, err = tx.Exec(ctx, `
			INSERT INTO admin_write_idempotency (xa_id, request_id, action, payload_hash, status_code, response_json)
			VALUES ($1,$2,$3,$4,0,'{}'::jsonb)
			ON CONFLICT (xa_id, request_id, action) DO NOTHING`, xaID, requestID, action, hash)
		if err != nil {
			return nil, false, err
		}
		if tag.RowsAffected() == 1 {
			return nil, false, nil
		}
		err = tx.QueryRow(ctx, `
			SELECT payload_hash, response_json
			FROM admin_write_idempotency
			WHERE xa_id = $1 AND request_id = $2 AND action = $3
			FOR UPDATE`, xaID, requestID, action).Scan(&got, &raw)
	}
	if err != nil {
		return nil, false, err
	}
	if got != hash {
		return nil, false, index.ErrIdempotency
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

func writeAudit(ctx context.Context, tx pgx.Tx, requestID, actor uuid.UUID, action string, documentID uuid.UUID, payload map[string]any) error {
	raw, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	_, err = tx.Exec(ctx, `
		INSERT INTO audit_logs (request_id, actor_user_id, action, entity_type, entity_id, payload)
		VALUES ($1,$2,$3,'document',$4,$5::jsonb)`, requestID, actor, action, documentID.String(), raw)
	return err
}
