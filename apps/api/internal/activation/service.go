package activation

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"strings"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
)

var versionPattern = regexp.MustCompile(`^[0-9]+\.[0-9]+\.[0-9]+$`)

type Service struct {
	Store *Store
	XAID  string
}

func (s *Service) ListProcedures(ctx context.Context) ([]ProcedureSummary, error) {
	return listProcedures(ctx, s.Store.Pool, s.XAID)
}

func (s *Service) ListVersions(ctx context.Context, procedureID uuid.UUID) ([]VersionSummary, error) {
	if _, err := s.mustProcedure(ctx, procedureID); err != nil {
		return nil, err
	}
	return listVersions(ctx, s.Store.Pool, s.XAID, procedureID)
}

func (s *Service) GetVersion(ctx context.Context, procedureID, versionID uuid.UUID) (VersionSummary, error) {
	tx, err := s.Store.begin(ctx)
	if err != nil {
		return VersionSummary{}, err
	}
	defer tx.Rollback(ctx)
	if _, _, _, _, _, err := lockProcedure(ctx, tx, s.XAID, procedureID); err != nil {
		return VersionSummary{}, err
	}
	row, err := loadVersion(ctx, tx, s.XAID, procedureID, versionID)
	if err != nil {
		return VersionSummary{}, err
	}
	sources, err := loadSources(ctx, tx, versionID)
	if err != nil {
		return VersionSummary{}, err
	}
	return toSummary(row, sources), nil
}

func (s *Service) History(ctx context.Context, procedureID uuid.UUID) ([]HistoryItem, error) {
	if _, err := s.mustProcedure(ctx, procedureID); err != nil {
		return nil, err
	}
	return listHistory(ctx, s.Store.Pool, procedureID)
}

func (s *Service) mustProcedure(ctx context.Context, procedureID uuid.UUID) (uuid.UUID, error) {
	var id uuid.UUID
	err := s.Store.Pool.QueryRow(ctx, `SELECT id FROM procedures WHERE id = $1 AND xa_id = $2`, procedureID, s.XAID).Scan(&id)
	if errors.Is(err, pgx.ErrNoRows) {
		return uuid.Nil, ErrNotFound
	}
	return id, err
}

func (s *Service) CreateVersion(ctx context.Context, actor, requestID, procedureID uuid.UUID, in CreateVersionRequest) (CreateVersionResult, error) {
	if !in.Confirm {
		return CreateVersionResult{}, fmt.Errorf("%w: confirm is required", ErrValidation)
	}
	hash := hashPayload(procedureID.String(), strings.TrimSpace(in.Version), string(in.Definition), uuidString(in.CopyFromVersionID))
	tx, err := s.Store.begin(ctx)
	if err != nil {
		return CreateVersionResult{}, err
	}
	defer tx.Rollback(ctx)
	replay, ok, err := claimIdempotency(ctx, tx, s.XAID, requestID, ActionCreate, hash)
	if err != nil {
		return CreateVersionResult{}, err
	}
	if ok {
		var out CreateVersionResult
		if err := json.Unmarshal(replay, &out); err != nil {
			return CreateVersionResult{}, err
		}
		out.Replayed = true
		return out, nil
	}
	_, code, name, domain, activeID, err := lockProcedure(ctx, tx, s.XAID, procedureID)
	if err != nil {
		return CreateVersionResult{}, err
	}
	sourceID := in.CopyFromVersionID
	if sourceID == nil {
		sourceID = activeID
	}
	raw := in.Definition
	if len(strings.TrimSpace(string(raw))) == 0 {
		if sourceID == nil {
			return CreateVersionResult{}, fmt.Errorf("%w: definition or copy_from_version_id is required", ErrValidation)
		}
		src, err := loadVersion(ctx, tx, s.XAID, procedureID, *sourceID)
		if err != nil {
			return CreateVersionResult{}, err
		}
		raw = src.Definition
	}
	label := strings.TrimSpace(in.Version)
	if label == "" {
		label, err = nextVersionLabel(ctx, tx, procedureID)
		if err != nil {
			return CreateVersionResult{}, err
		}
	}
	if !versionPattern.MatchString(label) {
		return CreateVersionResult{}, fmt.Errorf("%w: version must be n.n.n", ErrValidation)
	}
	body, err := wrapDefinition(raw, code, name, domain, s.XAID, label, "APPROVED")
	if err != nil {
		return CreateVersionResult{}, err
	}
	var id uuid.UUID
	err = tx.QueryRow(ctx, `
		INSERT INTO procedure_versions (
			procedure_id, version, status, definition, created_by, approved_by, approved_at
		) VALUES ($1,$2,'APPROVED',$3::jsonb,$4,$4,now())
		RETURNING id`, procedureID, label, body, actor).Scan(&id)
	if isUniqueViolation(err) {
		return CreateVersionResult{}, ErrDuplicateVersion
	}
	if err != nil {
		return CreateVersionResult{}, err
	}
	out := CreateVersionResult{ID: id, ProcedureID: procedureID, Version: label, Status: "APPROVED"}
	rawOut, err := json.Marshal(out)
	if err != nil {
		return CreateVersionResult{}, err
	}
	if err := writeAudit(ctx, tx, requestID, actor, AuditCreate, "procedure_version", id.String(), map[string]any{
		"xa_id": s.XAID, "procedure_id": procedureID.String(), "version": label,
	}); err != nil {
		return CreateVersionResult{}, err
	}
	if err := storeIdempotency(ctx, tx, s.XAID, requestID, ActionCreate, hash, 201, rawOut); err != nil {
		return CreateVersionResult{}, err
	}
	if err := tx.Commit(ctx); err != nil {
		return CreateVersionResult{}, err
	}
	return out, nil
}

func (s *Service) Activate(ctx context.Context, actor, requestID, procedureID, versionID uuid.UUID, expected *uuid.UUID) (ActivateResult, error) {
	hash := hashPayload(procedureID.String(), versionID.String(), uuidString(expected))
	tx, err := s.Store.begin(ctx)
	if err != nil {
		return ActivateResult{}, err
	}
	defer tx.Rollback(ctx)
	replay, ok, err := claimIdempotency(ctx, tx, s.XAID, requestID, ActionActivate, hash)
	if err != nil {
		return ActivateResult{}, err
	}
	if ok {
		var out ActivateResult
		if err := json.Unmarshal(replay, &out); err != nil {
			return ActivateResult{}, err
		}
		out.Replayed = true
		return out, nil
	}
	_, _, _, _, currentID, err := lockProcedure(ctx, tx, s.XAID, procedureID)
	if err != nil {
		return ActivateResult{}, err
	}
	if err := checkExpected(currentID, expected); err != nil {
		return ActivateResult{}, err
	}
	row, err := loadVersion(ctx, tx, s.XAID, procedureID, versionID)
	if err != nil {
		return ActivateResult{}, err
	}
	if row.XaID != s.XAID {
		return ActivateResult{}, ErrNotFound
	}
	if row.Status == "ACTIVE" {
		if currentID == nil || *currentID != row.ID {
			if _, err := tx.Exec(ctx, `
				UPDATE procedures SET active_version_id = $2, updated_at = now()
				WHERE id = $1 AND xa_id = $3`, procedureID, row.ID, s.XAID); err != nil {
				return ActivateResult{}, err
			}
		}
		out := ActivateResult{
			ProcedureID: procedureID, PreviousActiveVersionID: currentID, ActiveVersionID: row.ID,
			Version: row.Version, Status: "ACTIVE", ActivatedBy: actor, Noop: true,
		}
		if row.ActivatedAt != nil {
			out.ActivatedAt = *row.ActivatedAt
		}
		if row.ActivatedBy != nil {
			out.ActivatedBy = *row.ActivatedBy
		}
		rawOut, err := json.Marshal(out)
		if err != nil {
			return ActivateResult{}, err
		}
		if err := storeIdempotency(ctx, tx, s.XAID, requestID, ActionActivate, hash, 200, rawOut); err != nil {
			return ActivateResult{}, err
		}
		if err := tx.Commit(ctx); err != nil {
			return ActivateResult{}, err
		}
		return out, nil
	}
	sources, err := loadSources(ctx, tx, versionID)
	if err != nil {
		return ActivateResult{}, err
	}
	el := evaluateEligibility(row, sources)
	if !el.CanActivate {
		return ActivateResult{}, fmt.Errorf("%w: %s", mapReason(el), firstReason(el))
	}
	now := nowUTC()
	liveID, err := loadLiveActive(ctx, tx, procedureID)
	if err != nil {
		return ActivateResult{}, err
	}
	if _, err := tx.Exec(ctx, `
		UPDATE procedure_versions SET status = 'ARCHIVED'
		WHERE procedure_id = $1 AND status = 'ACTIVE' AND id <> $2`, procedureID, row.ID); err != nil {
		return ActivateResult{}, err
	}
	tag, err := tx.Exec(ctx, `
		UPDATE procedure_versions
		SET status = 'ACTIVE', activated_at = $3, activated_by = $4
		WHERE id = $1 AND procedure_id = $2 AND status IN ('APPROVED', 'INDEXING')`,
		row.ID, procedureID, now, actor)
	if err != nil {
		if isUniqueViolation(err) {
			return ActivateResult{}, ErrConflict
		}
		return ActivateResult{}, err
	}
	if tag.RowsAffected() != 1 {
		return ActivateResult{}, ErrConflict
	}
	if _, err := tx.Exec(ctx, `
		UPDATE procedures SET active_version_id = $2, updated_at = now()
		WHERE id = $1 AND xa_id = $3`, procedureID, row.ID, s.XAID); err != nil {
		return ActivateResult{}, err
	}
	previous := liveID
	if previous == nil {
		previous = currentID
	}
	out := ActivateResult{
		ProcedureID: procedureID, PreviousActiveVersionID: previous, ActiveVersionID: row.ID,
		Version: row.Version, Status: "ACTIVE", ActivatedAt: now, ActivatedBy: actor,
	}
	if err := writeAudit(ctx, tx, requestID, actor, AuditAction, "procedure", procedureID.String(), map[string]any{
		"xa_id":                      s.XAID,
		"procedure_id":               procedureID.String(),
		"previous_active_version_id": uuidString(currentID),
		"new_active_version_id":      row.ID.String(),
		"version":                    row.Version,
	}); err != nil {
		return ActivateResult{}, err
	}
	rawOut, err := json.Marshal(out)
	if err != nil {
		return ActivateResult{}, err
	}
	if err := storeIdempotency(ctx, tx, s.XAID, requestID, ActionActivate, hash, 200, rawOut); err != nil {
		return ActivateResult{}, err
	}
	if err := tx.Commit(ctx); err != nil {
		return ActivateResult{}, err
	}
	return out, nil
}

func checkExpected(current *uuid.UUID, expected *uuid.UUID) error {
	if expected == nil && current == nil {
		return nil
	}
	if expected == nil || current == nil {
		return ErrStale
	}
	if *expected != *current {
		return ErrStale
	}
	return nil
}

func uuidString(id *uuid.UUID) string {
	if id == nil {
		return ""
	}
	return id.String()
}

func mapReason(el Eligibility) error {
	switch firstReason(el) {
	case "INVALID_DEFINITION":
		return ErrInvalidDefinition
	case "DEFINITION_IDENTITY_MISMATCH":
		return ErrNotEligible
	case "VERSION_NOT_APPROVED", "VERSION_ARCHIVED", "VERSION_ALREADY_ACTIVE":
		return ErrNotApproved
	default:
		return ErrNotEligible
	}
}
