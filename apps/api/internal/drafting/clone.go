package drafting

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/google/uuid"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/repository"
)

// CloneDraft starts a new DRAFT from an approved or published one, so an Admin
// can correct a Definition already in use (e.g. a question that is never
// asked) without drafting again with AI. The copy keeps the Definition, the
// source scope, evidence and its reviews (same content), conflicts and
// coverage decisions; approval and publication are cleared and it is
// validated again. One open copy per draft: asking again returns it.
func (s *Service) CloneDraft(ctx context.Context, actor, draftID uuid.UUID) (DraftView, error) {
	row, err := s.Drafts.Get(ctx, s.XAID, draftID)
	if err != nil {
		return DraftView{}, err
	}
	if row.Status != repository.DraftStatusApproved && row.Status != repository.DraftStatusPublished {
		return DraftView{}, fmt.Errorf("%w: only an approved or published draft can be copied; edit this one directly", ErrReviewInvalid)
	}
	siblings, err := s.Drafts.ListByProcedure(ctx, s.XAID, row.ProcedureID)
	if err != nil {
		return DraftView{}, err
	}
	for _, sib := range siblings {
		if sib.Status != repository.DraftStatusDraft {
			continue
		}
		if v, err := viewFromRow(sib); err == nil && v.ParentDraftID != nil && *v.ParentDraftID == row.ID {
			return v, nil
		}
	}
	view, err := viewFromRow(row)
	if err != nil {
		return DraftView{}, err
	}
	parent := row.ID
	view.ParentDraftID = &parent
	view.ApprovedBy, view.ApprovedAt = nil, nil
	view.PublishedVersionID, view.PublishedVersion = nil, ""
	view.IndexRequestID, view.IndexJobID, view.JobID = nil, nil, nil
	view.Revision = 0
	view.Validation.SemanticApproved = false
	snap, err := s.resolveViewSource(ctx, view)
	if err != nil {
		return DraftView{}, err
	}
	view.Evidence = invalidateStaleEvidence(view.Evidence, view.Definition)
	revalidate(&view, snap, row.ProcedureCode, view.OriginalProcedureCode)
	payload, err := json.Marshal(storedPayloadFromView(view))
	if err != nil {
		return DraftView{}, err
	}
	validationRaw, err := json.Marshal(view.Validation)
	if err != nil {
		return DraftView{}, err
	}
	saved, err := s.Drafts.Insert(ctx, repository.ProcedureDraft{
		ID: uuid.New(), DocumentID: row.DocumentID, ProcedureID: row.ProcedureID,
		DraftDefinition: payload, ValidationResult: validationRaw,
		Status: repository.DraftStatusDraft, CreatedBy: &actor, UpdatedBy: &actor,
	})
	if err != nil {
		return DraftView{}, err
	}
	return viewFromRow(saved)
}
