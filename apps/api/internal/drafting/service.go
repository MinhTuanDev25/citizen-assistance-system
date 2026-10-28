package drafting

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/activation"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/decision"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/index"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/repository"
)

// DraftRepository is the persistence surface used by drafting. Tests wrap
// *repository.DraftRepo to inject UpdateStatus failures without production hooks.
type DraftRepository interface {
	Insert(ctx context.Context, row repository.ProcedureDraft) (repository.ProcedureDraft, error)
	Get(ctx context.Context, xaID string, draftID uuid.UUID) (repository.ProcedureDraft, error)
	ListByProcedure(ctx context.Context, xaID string, procedureID uuid.UUID) ([]repository.ProcedureDraft, error)
	ListAll(ctx context.Context, xaID string, procedureID *uuid.UUID) ([]repository.ProcedureDraft, error)
	UpdateDefinition(ctx context.Context, xaID string, draftID, actor uuid.UUID, definition, validation json.RawMessage, expectedUpdatedAt time.Time) (repository.ProcedureDraft, error)
	UpdateStatus(ctx context.Context, xaID string, draftID, actor uuid.UUID, status string, definition, validation json.RawMessage, expectedUpdatedAt time.Time, allowedFrom []string) (repository.ProcedureDraft, error)
}

type Service struct {
	XAID       string
	Source     SourceResolver
	Drafts     DraftRepository
	Client     *Client
	Activation *activation.Service
	Index      Indexer
	// Jobs/Runner enable multi-batch drafting of long documents.
	Jobs   JobStore
	Runner *JobRunner
}

// SourceResolver loads verified excerpts (*SourceStore). Tests inject doubles.
type SourceResolver interface {
	Resolve(ctx context.Context, xaID string, procedureID, documentID, generationID uuid.UUID, pageStart, pageEnd int) (SourceSnapshot, error)
	ResolveChunks(ctx context.Context, xaID string, procedureID, documentID, generationID uuid.UUID, ranges []ChunkRange) (SourceSnapshot, error)
	ListReady(ctx context.Context, xaID string, procedureID uuid.UUID) ([]SourceOption, error)
}

// Indexer is the persisted Link/Request boundary used during activation.
// *index.Service satisfies it. Tests inject deterministic doubles.
type Indexer interface {
	Link(ctx context.Context, in index.LinkInput) (index.Link, bool, error)
	Request(ctx context.Context, documentID, versionID, actor, requestID uuid.UUID, retry bool) (index.Result, error)
}

var (
	ErrUnauthorizedScope = errors.New("drafting: unauthorized scope")
	ErrNotConfigured     = errors.New("drafting: ai service not configured")
	ErrConflict          = errors.New("drafting: revision conflict")
	ErrApprovalBlocked   = errors.New("drafting: approval blocked")
	ErrNotApproved       = errors.New("drafting: draft is not approved")
	ErrAlreadyPublished  = errors.New("drafting: draft already published")
	// ErrReviewInvalid is a review decision the draft cannot take (no evidence, stale, invalid quote).
	ErrReviewInvalid = errors.New("drafting: review decision rejected")
)

func (s *Service) Generate(ctx context.Context, actor, requestID, procedureID uuid.UUID, req GenerateRequest) (DraftView, error) {
	if s == nil || s.Source == nil || s.Drafts == nil {
		return DraftView{}, ErrNotConfigured
	}
	if s.Client == nil || s.Client.BaseURL == "" {
		return DraftView{}, ErrNotConfigured
	}
	if req.DocumentID == uuid.Nil {
		return DraftView{}, fmt.Errorf("drafting: document_id is required")
	}
	snap, err := s.Source.Resolve(ctx, s.XAID, procedureID, req.DocumentID, req.GenerationID, req.SourcePageStart, req.SourcePageEnd)
	if err != nil {
		return DraftView{}, err
	}
	if err := checkDraftSize(snap); err != nil {
		return DraftView{}, err
	}
	provider, err := s.Client.Draft(ctx, requestID, snap)
	if err != nil {
		return DraftView{}, err
	}
	view, payload, validationRaw, err := s.buildView(snap, actor, provider)
	if err != nil {
		return DraftView{}, err
	}
	row := repository.ProcedureDraft{
		DocumentID:       snap.DocumentID,
		ProcedureID:      procedureID,
		DraftDefinition:  payload,
		ValidationResult: validationRaw,
		Status:           repository.DraftStatusDraft,
		CreatedBy:        &actor,
		UpdatedBy:        &actor,
	}
	saved, err := s.Drafts.Insert(ctx, row)
	if err != nil {
		return DraftView{}, err
	}
	view.ID = saved.ID
	view.Status = saved.Status
	view.CreatedAt = saved.CreatedAt
	view.UpdatedAt = saved.UpdatedAt
	view.CreatedBy = saved.CreatedBy
	view.UpdatedBy = saved.UpdatedBy
	return view, nil
}

func (s *Service) Get(ctx context.Context, draftID uuid.UUID) (DraftView, error) {
	row, err := s.Drafts.Get(ctx, s.XAID, draftID)
	if err != nil {
		return DraftView{}, err
	}
	return viewFromRow(row)
}

func (s *Service) List(ctx context.Context, procedureID uuid.UUID) ([]DraftView, error) {
	rows, err := s.Drafts.ListByProcedure(ctx, s.XAID, procedureID)
	if err != nil {
		return nil, err
	}
	return viewsFromRows(rows)
}

func (s *Service) ListAll(ctx context.Context, procedureID *uuid.UUID) ([]DraftView, error) {
	rows, err := s.Drafts.ListAll(ctx, s.XAID, procedureID)
	if err != nil {
		return nil, err
	}
	return viewsFromRows(rows)
}

func (s *Service) ListSources(ctx context.Context, procedureID uuid.UUID) ([]SourceOption, error) {
	if s.Source == nil {
		return nil, ErrNotConfigured
	}
	return s.Source.ListReady(ctx, s.XAID, procedureID)
}

func (s *Service) Validate(ctx context.Context, draftID uuid.UUID) (DraftView, error) {
	row, err := s.Drafts.Get(ctx, s.XAID, draftID)
	if err != nil {
		return DraftView{}, err
	}
	view, err := viewFromRow(row)
	if err != nil {
		return DraftView{}, err
	}
	snap, err := s.resolveViewSource(ctx, view)
	if err != nil {
		return DraftView{}, err
	}
	revalidate(&view, snap, row.ProcedureCode, view.OriginalProcedureCode)
	return view, nil
}

func (s *Service) Patch(ctx context.Context, actor, draftID uuid.UUID, req PatchRequest) (DraftView, error) {
	if req.ExpectedUpdatedAt == nil {
		return DraftView{}, fmt.Errorf("drafting: expected_updated_at is required")
	}
	row, err := s.Drafts.Get(ctx, s.XAID, draftID)
	if err != nil {
		return DraftView{}, err
	}
	if row.Status != repository.DraftStatusDraft {
		return DraftView{}, fmt.Errorf("drafting: only DRAFT rows can be patched")
	}
	view, err := viewFromRow(row)
	if err != nil {
		return DraftView{}, err
	}
	snap, err := s.resolveViewSource(ctx, view)
	if err != nil {
		return DraftView{}, err
	}
	locked, originalCode, mismatch, lockErr := lockProcedureIdentity(req.Definition, row.ProcedureCode, snap.ProcedureName)
	if lockErr != nil && locked == nil {
		return DraftView{}, lockErr
	}
	view.Evidence = followEditedProposals(view.Evidence, view.Definition, locked)
	view.Definition = locked
	view.IdentityMismatch = mismatch || view.IdentityMismatch
	if view.OriginalProcedureCode == "" && originalCode != "" {
		view.OriginalProcedureCode = originalCode
	}
	view.Evidence = invalidateStaleEvidence(view.Evidence, locked)
	revalidate(&view, snap, row.ProcedureCode, view.OriginalProcedureCode)
	applyIdentityFinding(&view.Validation, originalCode, row.ProcedureCode, mismatch)
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

func (s *Service) Review(ctx context.Context, actor, draftID uuid.UUID, req ReviewRequest) (DraftView, error) {
	if req.ExpectedUpdatedAt == nil {
		return DraftView{}, fmt.Errorf("drafting: expected_updated_at is required")
	}
	row, err := s.Drafts.Get(ctx, s.XAID, draftID)
	if err != nil {
		return DraftView{}, err
	}
	if row.Status != repository.DraftStatusDraft {
		return DraftView{}, fmt.Errorf("drafting: only DRAFT rows can be reviewed")
	}
	view, err := viewFromRow(row)
	if err != nil {
		return DraftView{}, err
	}
	snap, err := s.resolveViewSource(ctx, view)
	if err != nil {
		return DraftView{}, err
	}
	if err := applyReviewDecisions(&view, req.Decisions); err != nil {
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

func (s *Service) Approve(ctx context.Context, actor, draftID uuid.UUID, req ApproveRequest) (DraftView, error) {
	if !req.Confirm {
		return DraftView{}, fmt.Errorf("drafting: confirm is required")
	}
	if req.ExpectedUpdatedAt == nil {
		return DraftView{}, fmt.Errorf("drafting: expected_updated_at is required")
	}
	row, err := s.Drafts.Get(ctx, s.XAID, draftID)
	if err != nil {
		return DraftView{}, err
	}
	if row.Status == repository.DraftStatusApproved || row.Status == repository.DraftStatusPublished {
		view, vErr := viewFromRow(row)
		if vErr != nil {
			return DraftView{}, vErr
		}
		if req.ExpectedUpdatedAt.Equal(row.UpdatedAt) {
			return view, nil
		}
		return DraftView{}, ErrConflict
	}
	if row.Status != repository.DraftStatusDraft && row.Status != repository.DraftStatusReviewed {
		return DraftView{}, fmt.Errorf("drafting: draft status %s cannot be approved", row.Status)
	}
	view, err := viewFromRow(row)
	if err != nil {
		return DraftView{}, err
	}
	snap, err := s.resolveViewSource(ctx, view)
	if err != nil {
		return DraftView{}, err
	}
	revalidate(&view, snap, row.ProcedureCode, view.OriginalProcedureCode)
	if err := approvalGate(view); err != nil {
		return view, err
	}
	now := time.Now().UTC()
	view.ApprovedBy = &actor
	view.ApprovedAt = &now
	view.Validation.SemanticApproved = true
	view.Validation.Readiness = "admin_approved"
	view.Status = repository.DraftStatusApproved
	payload, err := json.Marshal(storedPayloadFromView(view))
	if err != nil {
		return DraftView{}, err
	}
	validationRaw, err := json.Marshal(view.Validation)
	if err != nil {
		return DraftView{}, err
	}
	saved, err := s.Drafts.UpdateStatus(ctx, s.XAID, draftID, actor, repository.DraftStatusApproved, payload, validationRaw, *req.ExpectedUpdatedAt, []string{repository.DraftStatusDraft, repository.DraftStatusReviewed})
	if err != nil {
		if errors.Is(err, repository.ErrNotFound) {
			return DraftView{}, ErrConflict
		}
		return DraftView{}, err
	}
	return viewFromRow(saved)
}

func (s *Service) buildView(snap SourceSnapshot, actor uuid.UUID, provider *ProviderResponse) (DraftView, json.RawMessage, json.RawMessage, error) {
	originalCode := strings.TrimSpace(provider.OriginalProcedureCode)
	if originalCode == "" {
		originalCode = strings.TrimSpace(provider.ProcedureCode)
	}
	locked, lockedOriginal, mismatch, _ := lockProcedureIdentity(provider.Definition, snap.ProcedureCode, snap.ProcedureName)
	if lockedOriginal != "" {
		originalCode = lockedOriginal
	}
	if provider.IdentityMismatch {
		mismatch = true
	}
	goVal := decision.ValidateDraftDefinition(locked, snap.ProcedureCode)
	cov := EvaluateCoverage(locked, provider.Evidence, snap.Excerpts, snap)
	applyCoverage(&goVal, cov)
	applyIdentityFinding(&goVal, originalCode, snap.ProcedureCode, mismatch)
	note := ""
	if snap.GenerationPVStatus != "ACTIVE" {
		note = fmt.Sprintf(
			"source generation %s belongs to procedure version %s (%s); it is drafting INPUT only and is not copied onto a new version",
			snap.GenerationID, snap.GenerationVersion, snap.GenerationPVStatus,
		)
	}
	view := DraftView{
		ProcedureID:           snap.ProcedureID,
		ProcedureCode:         snap.ProcedureCode,
		DocumentID:            snap.DocumentID,
		Status:                repository.DraftStatusDraft,
		Definition:            locked,
		Evidence:              cov.Evidence,
		Proposals:             cov.Proposals,
		Warnings:              append([]string{}, provider.Warnings...),
		MissingRequirements:   append([]string{}, provider.MissingRequirements...),
		Validation:            goVal,
		Model:                 provider.Model,
		Provider:              provider.Provider,
		PromptVersion:         provider.PromptVersion,
		PromptHash:            provider.PromptHash,
		GenerationID:          snap.GenerationID,
		SourcePageStart:       snap.SourcePageStart,
		SourcePageEnd:         snap.SourcePageEnd,
		Usage:                 provider.Usage,
		LatencyMS:             provider.LatencyMS,
		CreatedBy:             &actor,
		HistoricalNote:        note,
		IdentityMismatch:      mismatch,
		OriginalProcedureCode: originalCode,
		Revision:              1,
	}
	if view.Warnings == nil {
		view.Warnings = []string{}
	}
	payload, err := json.Marshal(storedPayloadFromView(view))
	if err != nil {
		return DraftView{}, nil, nil, err
	}
	validationRaw, err := json.Marshal(goVal)
	if err != nil {
		return DraftView{}, nil, nil, err
	}
	return view, payload, validationRaw, nil
}

func (s *Service) resolveViewSource(ctx context.Context, view DraftView) (SourceSnapshot, error) {
	if len(view.ChunkRanges) > 0 {
		return s.Source.ResolveChunks(ctx, s.XAID, view.ProcedureID, view.DocumentID, view.GenerationID, view.ChunkRanges)
	}
	return s.Source.Resolve(ctx, s.XAID, view.ProcedureID, view.DocumentID, view.GenerationID, view.SourcePageStart, view.SourcePageEnd)
}

func revalidate(view *DraftView, snap SourceSnapshot, expectedCode, originalCode string) {
	if originalCode == "" {
		originalCode = view.OriginalProcedureCode
	}
	goVal := decision.ValidateDraftDefinition(view.Definition, expectedCode)
	view.AdminDecisions = currentDecisions(view.Definition, view.AdminDecisions)
	cov := EvaluateCoverage(view.Definition, view.Evidence, snap.Excerpts, snap, decidedContent(view.AdminDecisions))
	preserveReviews(cov.Evidence, view.Evidence)
	applyCoverage(&goVal, cov)
	if originalCode != "" {
		applyIdentityFinding(&goVal, originalCode, expectedCode, originalCode != expectedCode)
		view.OriginalProcedureCode = originalCode
	} else if view.IdentityMismatch {
		applyIdentityFinding(&goVal, "", expectedCode, true)
	}
	view.Evidence = cov.Evidence
	view.Proposals = cov.Proposals
	view.Validation = goVal
	view.Validation.SemanticApproved = false
	applyAggregateFindings(view, snap.Excerpts)
}

func preserveReviews(next, prev []EvidenceItem) {
	byID := map[string]EvidenceItem{}
	for _, item := range prev {
		byID[item.ProposalID] = item
	}
	for i := range next {
		old, ok := byID[next[i].ProposalID]
		if !ok {
			continue
		}
		if next[i].ReviewStatus == ReviewStale || next[i].ReviewStatus == ReviewUnknownProposal {
			continue
		}
		if old.BoundFingerprint != "" && old.BoundFingerprint == next[i].BoundFingerprint && old.ReviewStatus != "" {
			next[i].ReviewStatus = old.ReviewStatus
		}
	}
}

// followEditedProposals keeps evidence attached to its proposal across an
// Admin edit. List proposals are numbered (checklist:3), so removing one moves
// the later ones: their evidence follows the same content to its new number.
// Evidence of a proposal the Admin removed is dropped, as is evidence that
// already pointed at nothing. Content edited in place keeps its evidence,
// which invalidateStaleEvidence then marks stale.
func followEditedProposals(items []EvidenceItem, before, after json.RawMessage) []EvidenceItem {
	oldProps, errOld := EnumerateMaterialProposals(before)
	newProps, errNew := EnumerateMaterialProposals(after)
	if errOld != nil || errNew != nil {
		return items
	}
	key := func(p MaterialProposal) string { return p.Type + "\n" + p.Content }
	newByID := proposalsByID(newProps)
	target := map[string]string{} // old id -> new id ("" = removed)
	claimed := map[string]bool{}
	pending := make([]MaterialProposal, 0)
	for _, op := range oldProps {
		if np, ok := newByID[op.ID]; ok && key(np) == key(op) {
			target[op.ID], claimed[op.ID] = op.ID, true
			continue
		}
		pending = append(pending, op)
	}
	rest := pending[:0:0]
	for _, op := range pending {
		moved := false
		for _, np := range newProps {
			if !claimed[np.ID] && key(np) == key(op) {
				target[op.ID], claimed[np.ID], moved = np.ID, true, true
				break
			}
		}
		if !moved {
			rest = append(rest, op)
		}
	}
	for _, op := range rest {
		if _, ok := newByID[op.ID]; ok && !claimed[op.ID] {
			target[op.ID], claimed[op.ID] = op.ID, true // edited in place
		} else {
			target[op.ID] = "" // removed
		}
	}
	oldByID := proposalsByID(oldProps)
	out := make([]EvidenceItem, 0, len(items))
	for _, item := range items {
		id := item.ProposalID
		to, known := target[id]
		switch {
		case !known:
			if _, still := newByID[id]; !still {
				continue // pointed at nothing before and after the edit
			}
		case to == "":
			continue
		case to != id:
			if item.BoundFingerprint == oldByID[id].Fingerprint {
				item.BoundFingerprint = newByID[to].Fingerprint
			}
			item.ProposalID, item.RawProposalID = to, to
			item.ProposalFingerprint = newByID[to].Fingerprint
		}
		out = append(out, item)
	}
	return out
}

func invalidateStaleEvidence(items []EvidenceItem, definition []byte) []EvidenceItem {
	props, err := EnumerateMaterialProposals(definition)
	if err != nil {
		for i := range items {
			items[i].ReviewStatus = ReviewStale
		}
		return items
	}
	known := proposalsByID(props)
	out := make([]EvidenceItem, 0, len(items))
	for _, item := range items {
		norm := NormalizeProposalID(firstNonEmpty(item.RawProposalID, item.ProposalID), known)
		if norm.Normalized {
			item.RawProposalID = firstNonEmpty(item.RawProposalID, item.ProposalID)
			item.ProposalID = norm.Canonical
		}
		prop, ok := known[item.ProposalID]
		if !ok {
			item.ReviewStatus = ReviewStale
			item.QuoteValid = false
			out = append(out, item)
			continue
		}
		if item.BoundFingerprint != "" && item.BoundFingerprint != prop.Fingerprint {
			item.ReviewStatus = ReviewStale
			item.QuoteValid = false
		} else if item.BoundFingerprint == "" {
			item.BoundFingerprint = prop.Fingerprint
		}
		item.ProposalFingerprint = prop.Fingerprint
		out = append(out, item)
	}
	return out
}

func applyReviewDecisions(view *DraftView, decisions []ReviewDecision) error {
	byID := map[string]int{}
	for i, item := range view.Evidence {
		byID[item.ProposalID] = i
	}
	for _, decision := range decisions {
		pid := strings.TrimSpace(decision.ProposalID)
		status := strings.TrimSpace(decision.Decision)
		switch status {
		case ReviewAccepted, ReviewRejected, ReviewNeedsCorrection, ReviewPending:
		default:
			return fmt.Errorf("%w: invalid review decision %q", ErrReviewInvalid, status)
		}
		idx, ok := byID[pid]
		if !ok {
			return fmt.Errorf("%w: proposal %s has no source evidence to review", ErrReviewInvalid, pid)
		}
		if view.Evidence[idx].ReviewStatus == ReviewStale {
			return fmt.Errorf("%w: cannot accept stale evidence for %s", ErrReviewInvalid, pid)
		}
		if status == ReviewAccepted && !view.Evidence[idx].QuoteValid {
			return fmt.Errorf("%w: cannot accept invalid evidence for %s", ErrReviewInvalid, pid)
		}
		view.Evidence[idx].ReviewStatus = status
	}
	return nil
}

func approvalGate(view DraftView) error {
	if view.IdentityMismatch || view.Validation.IdentityMismatch {
		return fmt.Errorf("%w: procedure identity mismatch", ErrApprovalBlocked)
	}
	if !view.Validation.SchemaValid {
		return fmt.Errorf("%w: definition schema is invalid", ErrApprovalBlocked)
	}
	if !view.Validation.ProceduralReady {
		return fmt.Errorf("%w: definition is not procedurally usable", ErrApprovalBlocked)
	}
	if !view.Validation.SourceCoverageComplete || !view.Validation.SourceQuotesVerified || !view.Validation.GroundingVerified {
		return fmt.Errorf("%w: source coverage is incomplete", ErrApprovalBlocked)
	}
	if blockingErrors(view.Validation.Issues) {
		return fmt.Errorf("%w: unresolved blocking validation findings", ErrApprovalBlocked)
	}
	for _, c := range view.Conflicts {
		if c.Status != ConflictResolved && !c.Inactive {
			return fmt.Errorf("%w: source conflict on %s is not resolved", ErrApprovalBlocked, c.Field)
		}
	}
	if view.Validation.StaleEvidenceCount > 0 {
		return fmt.Errorf("%w: stale evidence remains", ErrApprovalBlocked)
	}
	accepted := map[string]bool{}
	for _, item := range view.Evidence {
		if item.ReviewStatus == ReviewAccepted && item.QuoteValid {
			accepted[item.ProposalID] = true
		}
	}
	decided := decidedContent(view.AdminDecisions)
	for _, prop := range view.Proposals {
		if decided[proposalContentKey(prop)] {
			accepted[prop.ID] = true
		}
	}
	for _, prop := range view.Proposals {
		if !accepted[prop.ID] {
			return fmt.Errorf("%w: material proposal %s is not Admin-accepted", ErrApprovalBlocked, prop.ID)
		}
	}
	return nil
}

func storedPayloadFromView(view DraftView) storedPayload {
	return storedPayload{
		SchemaVersion:         "draft_definition.v1",
		Definition:            view.Definition,
		Evidence:              view.Evidence,
		Warnings:              view.Warnings,
		MissingRequirements:   view.MissingRequirements,
		Model:                 view.Model,
		Provider:              view.Provider,
		PromptVersion:         view.PromptVersion,
		PromptHash:            view.PromptHash,
		GenerationID:          view.GenerationID,
		DocumentID:            view.DocumentID,
		SourcePageStart:       view.SourcePageStart,
		SourcePageEnd:         view.SourcePageEnd,
		Usage:                 view.Usage,
		LatencyMS:             view.LatencyMS,
		HistoricalNote:        view.HistoricalNote,
		OriginalProcedureCode: view.OriginalProcedureCode,
		IdentityMismatch:      view.IdentityMismatch,
		Revision:              view.Revision,
		ApprovedBy:            view.ApprovedBy,
		ApprovedAt:            view.ApprovedAt,
		PublishedVersionID:    view.PublishedVersionID,
		PublishedVersion:      view.PublishedVersion,
		IndexRequestID:        view.IndexRequestID,
		IndexJobID:            view.IndexJobID,
		JobID:                 view.JobID,
		ParentDraftID:         view.ParentDraftID,
		ChunkRanges:           view.ChunkRanges,
		Coverage:              view.Coverage,
		Conflicts:             view.Conflicts,
		Provenance:            view.Provenance,
		AdminDecisions:        view.AdminDecisions,
	}
}

func viewFromRow(row repository.ProcedureDraft) (DraftView, error) {
	var payload storedPayload
	if len(row.DraftDefinition) > 0 {
		if err := json.Unmarshal(row.DraftDefinition, &payload); err != nil {
			return DraftView{}, err
		}
	}
	var validation decision.DraftValidation
	if len(row.ValidationResult) > 0 {
		_ = json.Unmarshal(row.ValidationResult, &validation)
	}
	if row.Status != repository.DraftStatusApproved && row.Status != repository.DraftStatusPublished {
		validation.SemanticApproved = false
	}
	proposals, _ := EnumerateMaterialProposals(payload.Definition)
	view := DraftView{
		ID:                    row.ID,
		ProcedureID:           row.ProcedureID,
		ProcedureCode:         row.ProcedureCode,
		DocumentID:            row.DocumentID,
		Status:                row.Status,
		Definition:            payload.Definition,
		Evidence:              payload.Evidence,
		Proposals:             proposals,
		Warnings:              payload.Warnings,
		MissingRequirements:   payload.MissingRequirements,
		Validation:            validation,
		Model:                 payload.Model,
		Provider:              payload.Provider,
		PromptVersion:         payload.PromptVersion,
		PromptHash:            payload.PromptHash,
		GenerationID:          payload.GenerationID,
		SourcePageStart:       payload.SourcePageStart,
		SourcePageEnd:         payload.SourcePageEnd,
		Usage:                 payload.Usage,
		LatencyMS:             payload.LatencyMS,
		CreatedBy:             row.CreatedBy,
		UpdatedBy:             row.UpdatedBy,
		CreatedAt:             row.CreatedAt,
		UpdatedAt:             row.UpdatedAt,
		HistoricalNote:        payload.HistoricalNote,
		IdentityMismatch:      payload.IdentityMismatch,
		Revision:              payload.Revision,
		ApprovedBy:            payload.ApprovedBy,
		ApprovedAt:            payload.ApprovedAt,
		PublishedVersionID:    payload.PublishedVersionID,
		PublishedVersion:      payload.PublishedVersion,
		OriginalProcedureCode: payload.OriginalProcedureCode,
		IndexRequestID:        payload.IndexRequestID,
		IndexJobID:            payload.IndexJobID,
		JobID:                 payload.JobID,
		ParentDraftID:         payload.ParentDraftID,
		ChunkRanges:           payload.ChunkRanges,
		Coverage:              payload.Coverage,
		Conflicts:             payload.Conflicts,
		Provenance:            payload.Provenance,
		AdminDecisions:        payload.AdminDecisions,
	}
	// An editable draft is read with its conflicts checked against the current
	// Definition (e.g. a decision stored before decisions were pinned reopens)
	// and its SOURCE_CONFLICT issues matching them, so a read, the UI and the
	// approval gate never trust a stale RESOLVED.
	if row.Status == repository.DraftStatusDraft && len(view.Conflicts) > 0 {
		reconcileConflicts(&view)
		syncConflictIssues(&view)
	}
	return view, nil
}

func viewsFromRows(rows []repository.ProcedureDraft) ([]DraftView, error) {
	out := make([]DraftView, 0, len(rows))
	for _, row := range rows {
		view, err := viewFromRow(row)
		if err != nil {
			return nil, err
		}
		out = append(out, view)
	}
	return out, nil
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if strings.TrimSpace(v) != "" {
			return v
		}
	}
	return ""
}
