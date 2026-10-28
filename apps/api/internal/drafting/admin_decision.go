package drafting

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/repository"
)

func proposalContentKey(p MaterialProposal) string {
	return p.Type + "\n" + p.Content
}

func decidedContent(items []AdminDecision) map[string]bool {
	out := map[string]bool{}
	for _, d := range items {
		out[d.Content] = true
	}
	return out
}

// currentDecisions keeps the decisions whose proposal content still exists,
// updating the ID when the proposal moved (an earlier list item was removed).
func currentDecisions(definition json.RawMessage, items []AdminDecision) []AdminDecision {
	if len(items) == 0 {
		return items
	}
	props, err := EnumerateMaterialProposals(definition)
	if err != nil {
		return items
	}
	idByKey := map[string]string{}
	for _, p := range props {
		if _, ok := idByKey[proposalContentKey(p)]; !ok {
			idByKey[proposalContentKey(p)] = p.ID
		}
	}
	out := make([]AdminDecision, 0, len(items))
	for _, d := range items {
		if id, ok := idByKey[d.Content]; ok {
			d.ProposalID = id
			out = append(out, d)
		}
	}
	return out
}

// DecideWithoutSource records (or withdraws) an Admin decision to keep a
// proposal that has no valid source quotation. A reason is required; the
// proposal is then shown in review as "Quyết định của cán bộ". A proposal that
// has a valid quotation does not need one.
func (s *Service) DecideWithoutSource(ctx context.Context, actor, draftID uuid.UUID, req AdminDecisionRequest) (DraftView, error) {
	if req.ExpectedUpdatedAt == nil {
		return DraftView{}, fmt.Errorf("drafting: expected_updated_at is required")
	}
	reason := strings.TrimSpace(req.Reason)
	if !req.Remove && (len([]rune(reason)) < 10 || len([]rune(reason)) > 500) {
		return DraftView{}, fmt.Errorf("%w: reason must be 10-500 characters", ErrReviewInvalid)
	}
	row, err := s.Drafts.Get(ctx, s.XAID, draftID)
	if err != nil {
		return DraftView{}, err
	}
	if row.Status != repository.DraftStatusDraft {
		return DraftView{}, fmt.Errorf("%w: only DRAFT rows can be changed", ErrReviewInvalid)
	}
	view, err := viewFromRow(row)
	if err != nil {
		return DraftView{}, err
	}
	var prop *MaterialProposal
	for i := range view.Proposals {
		if view.Proposals[i].ID == req.ProposalID {
			prop = &view.Proposals[i]
		}
	}
	if prop == nil {
		return DraftView{}, fmt.Errorf("%w: unknown proposal %s", ErrReviewInvalid, req.ProposalID)
	}
	key := proposalContentKey(*prop)
	kept := view.AdminDecisions[:0:0]
	for _, d := range view.AdminDecisions {
		if d.Content != key {
			kept = append(kept, d)
		}
	}
	if !req.Remove {
		for _, ev := range view.Evidence {
			if ev.ProposalID == prop.ID && ev.QuoteValid {
				return DraftView{}, fmt.Errorf("%w: this proposal has a valid source quotation; accept it instead", ErrReviewInvalid)
			}
		}
		kept = append(kept, AdminDecision{ProposalID: prop.ID, Content: key, Reason: reason, By: actor, At: time.Now().UTC()})
	}
	view.AdminDecisions = kept
	snap, err := s.resolveViewSource(ctx, view)
	if err != nil {
		return DraftView{}, err
	}
	revalidate(&view, snap, row.ProcedureCode, view.OriginalProcedureCode)
	payload, err := json.Marshal(storedPayloadFromView(view))
	if err != nil {
		return DraftView{}, err
	}
	validationRaw, err := json.Marshal(view.Validation)
	if err != nil {
		return DraftView{}, err
	}
	saved, err := s.Drafts.UpdateDefinition(ctx, s.XAID, draftID, actor, payload, validationRaw, *req.ExpectedUpdatedAt)
	if err != nil {
		if errors.Is(err, repository.ErrNotFound) {
			return DraftView{}, ErrConflict
		}
		return DraftView{}, err
	}
	return viewFromRow(saved)
}
