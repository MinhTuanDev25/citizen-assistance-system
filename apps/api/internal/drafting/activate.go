package drafting

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/activation"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/index"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/repository"
)

const (
	PhaseApproved               = "APPROVED"
	PhaseVersionCreated         = "VERSION_CREATED"
	PhaseIndexingPending        = "INDEXING_PENDING"
	PhaseIndexingRunning        = "INDEXING_RUNNING"
	PhaseIndexingFailed         = "INDEXING_FAILED"
	PhaseSourceReady            = "SOURCE_READY"
	PhaseActivated              = "ACTIVATED"
	PhaseReconciliationRequired = "RECONCILIATION_REQUIRED"
)

type ActivateDraftResult struct {
	Draft                  DraftView                       `json:"draft"`
	Version                *activation.CreateVersionResult `json:"version,omitempty"`
	Index                  *index.Result                   `json:"index,omitempty"`
	Activation             *activation.ActivateResult      `json:"activation,omitempty"`
	Activated              bool                            `json:"activated"`
	CanActivate            bool                            `json:"can_activate"`
	IndexingPending        bool                            `json:"indexing_pending,omitempty"`
	ReconciliationRequired bool                            `json:"reconciliation_required,omitempty"`
	Phase                  string                          `json:"phase"`
	Reasons                []string                        `json:"reasons,omitempty"`
	BlockedReason          string                          `json:"blocked_reason,omitempty"`
	HistoricalG3Note       string                          `json:"historical_g3_note,omitempty"`
}

func (s *Service) Activate(ctx context.Context, actor, requestID, draftID uuid.UUID, req ActivateDraftRequest) (ActivateDraftResult, error) {
	out := ActivateDraftResult{
		Phase:            PhaseApproved,
		HistoricalG3Note: "historical G3 generations are never cloned or FK-moved; a new version-specific generation must be created through Link+Index",
	}
	if !req.Confirm {
		return out, fmt.Errorf("drafting: confirm is required")
	}
	if s.Activation == nil {
		return out, fmt.Errorf("%w: activation service is not configured", ErrNotConfigured)
	}
	row, err := s.Drafts.Get(ctx, s.XAID, draftID)
	if err != nil {
		return out, err
	}
	view, err := viewFromRow(row)
	if err != nil {
		return out, err
	}
	out.Draft = view
	if row.Status != repository.DraftStatusApproved && row.Status != repository.DraftStatusPublished {
		return out, ErrNotApproved
	}

	resuming := view.PublishedVersionID != nil
	if !resuming {
		if req.ExpectedUpdatedAt == nil {
			return out, fmt.Errorf("drafting: expected_updated_at is required")
		}
		if !req.ExpectedUpdatedAt.Equal(row.UpdatedAt) {
			return out, ErrConflict
		}
		if err := approvalGate(view); err != nil {
			out.Draft = view
			return out, err
		}
	}

	if view.PublishedVersionID != nil {
		if recon, handled, reconErr := s.reconcileIfAlreadyActive(ctx, actor, draftID, row, &view, &out); handled {
			return recon, reconErr
		}
	}

	versionID, err := s.ensureVersion(ctx, actor, draftID, req, &row, &view, &out)
	if err != nil {
		return out, err
	}

	if s.Index == nil {
		summary, getErr := s.Activation.GetVersion(ctx, view.ProcedureID, versionID)
		if getErr == nil && summary.Eligibility.CanActivate {
			out.Phase = PhaseSourceReady
			activated, actErr := s.Activation.Activate(ctx, actor, draftOpID(draftID, "activate-version"), view.ProcedureID, versionID, req.ExpectedActiveVersionID)
			if actErr != nil {
				out.BlockedReason = "ACTIVATE_FAILED"
				out.Reasons = []string{actErr.Error()}
				out.Draft = view
				return out, actErr
			}
			return s.finishActivation(ctx, actor, draftID, row, &view, &out, &activated)
		}
		out.Phase = PhaseVersionCreated
		out.BlockedReason = "INDEX_SERVICE_UNAVAILABLE"
		out.Reasons = []string{"NO_SOURCE_DOCUMENT"}
		out.Draft = view
		return out, nil
	}

	pageRange := strings.TrimSpace(req.PageRange)
	if pageRange == "" {
		pageRange = fmt.Sprintf("%d-%d", view.SourcePageStart, view.SourcePageEnd)
	}
	linkID := draftOpID(draftID, "link")
	if _, _, err := s.Index.Link(ctx, index.LinkInput{
		DocumentID:         view.DocumentID,
		ProcedureVersionID: versionID,
		RelationshipType:   "SOURCE",
		PageRange:          pageRange,
		ActorUserID:        actor,
		RequestID:          linkID,
	}); err != nil {
		out.Phase = PhaseVersionCreated
		out.BlockedReason = "DOCUMENT_LINK_FAILED"
		out.Reasons = []string{err.Error()}
		out.Draft = view
		return out, nil
	}

	summary, err := s.Activation.GetVersion(ctx, view.ProcedureID, versionID)
	if err != nil {
		out.BlockedReason = "ELIGIBILITY_CHECK_FAILED"
		out.Reasons = []string{err.Error()}
		return out, nil
	}
	srcState := sourceIndexState(summary)
	switch srcState {
	case "READY":
		out.Phase = PhaseSourceReady
	case "FAILED":
		out.Phase = PhaseIndexingFailed
		out.BlockedReason = "INDEXING_FAILED"
		out.Reasons = summary.Eligibility.Reasons
		out.Draft = view
		return out, nil
	case "PROCESSING":
		out.Phase = PhaseIndexingRunning
		out.IndexingPending = true
		out.CanActivate = false
		out.BlockedReason = "INDEXING_PENDING"
		out.Reasons = pendingReasons(summary)
		out.Draft = view
		return out, nil
	default:
		indexReqID := draftOpID(draftID, "index")
		view.IndexRequestID = &indexReqID
		idx, reqErr := s.Index.Request(ctx, view.DocumentID, versionID, actor, indexReqID, false)
		if reqErr != nil {
			if isIndexInProgress(reqErr) {
				out.Phase = PhaseIndexingRunning
				out.IndexingPending = true
				out.CanActivate = false
				out.BlockedReason = "INDEXING_PENDING"
				out.Reasons = []string{reqErr.Error()}
				out.Draft = view
				return out, nil
			}
			out.Phase = PhaseIndexingFailed
			out.BlockedReason = "INDEX_REQUEST_FAILED"
			out.Reasons = []string{reqErr.Error()}
			out.Draft = view
			return out, nil
		}
		out.Index = &idx
		if idx.JobID != uuid.Nil {
			jobID := idx.JobID
			view.IndexJobID = &jobID
		}
		savedMeta, persistErr := s.persistApprovedPayload(ctx, actor, draftID, row.UpdatedAt, view)
		if persistErr != nil {
			return s.metadataPersistFailure(ctx, out, view, idx, persistErr)
		} else if savedMeta != nil {
			view = *savedMeta
			row.UpdatedAt = savedMeta.UpdatedAt
			out.Draft = view
		}
		if strings.EqualFold(idx.LinkStatus, "READY") {
			out.Phase = PhaseSourceReady
		} else if strings.EqualFold(idx.LinkStatus, "FAILED") || strings.EqualFold(idx.JobStatus, "FAILED") {
			out.Phase = PhaseIndexingFailed
			out.BlockedReason = firstNonEmpty(idx.ErrorCode, "INDEXING_FAILED")
			out.Draft = view
			return out, nil
		} else {
			out.Phase = PhaseIndexingRunning
			out.IndexingPending = true
			out.CanActivate = false
			out.BlockedReason = "INDEXING_PENDING"
			out.Reasons = []string{firstNonEmpty(idx.LinkStatus, "DOCUMENT_INDEX_PROCESSING")}
			out.Draft = view
			return out, nil
		}
		summary, err = s.Activation.GetVersion(ctx, view.ProcedureID, versionID)
		if err != nil {
			out.BlockedReason = "ELIGIBILITY_CHECK_FAILED"
			out.Reasons = []string{err.Error()}
			return out, nil
		}
	}

	el := summary.Eligibility
	out.CanActivate = el.CanActivate
	out.Reasons = el.Reasons
	if !el.CanActivate {
		if sourceIndexState(summary) == "PROCESSING" || hasReason(el.Reasons, "DOCUMENT_INDEX_PROCESSING") {
			out.Phase = PhaseIndexingRunning
			out.IndexingPending = true
			out.BlockedReason = "INDEXING_PENDING"
		} else {
			out.Phase = PhaseVersionCreated
			out.BlockedReason = firstNonEmpty(el.Reasons...)
		}
		out.Draft = view
		return out, nil
	}

	activateID := draftOpID(draftID, "activate-version")
	activated, err := s.Activation.Activate(ctx, actor, activateID, view.ProcedureID, versionID, req.ExpectedActiveVersionID)
	if err != nil {
		out.BlockedReason = "ACTIVATE_FAILED"
		out.Reasons = []string{err.Error()}
		out.Draft = view
		return out, err
	}
	return s.finishActivation(ctx, actor, draftID, row, &view, &out, &activated)
}

func (s *Service) finishActivation(ctx context.Context, actor, draftID uuid.UUID, row repository.ProcedureDraft, view *DraftView, out *ActivateDraftResult, activated *activation.ActivateResult) (ActivateDraftResult, error) {
	out.Activation = activated
	out.Activated = true
	out.Phase = PhaseActivated
	out.CanActivate = false
	out.BlockedReason = ""
	if pubErr := s.publishDraft(ctx, actor, draftID, row, view); pubErr != nil {
		out.ReconciliationRequired = true
		out.Phase = PhaseReconciliationRequired
		out.BlockedReason = "DRAFT_PUBLISH_PENDING"
		out.Reasons = []string{pubErr.Error()}
		out.Draft = *view
		return *out, nil
	}
	out.Draft = *view
	return *out, nil
}

func (s *Service) reconcileIfAlreadyActive(ctx context.Context, actor, draftID uuid.UUID, row repository.ProcedureDraft, view *DraftView, out *ActivateDraftResult) (ActivateDraftResult, bool, error) {
	if view.PublishedVersionID == nil {
		return *out, false, nil
	}
	summary, err := s.Activation.GetVersion(ctx, view.ProcedureID, *view.PublishedVersionID)
	if err != nil {
		return *out, false, nil
	}
	out.Version = &activation.CreateVersionResult{
		ID: summary.ID, ProcedureID: view.ProcedureID, Version: summary.Version, Status: summary.Status,
	}
	if summary.Status != "ACTIVE" {
		if row.Status == repository.DraftStatusPublished {
			out.Phase = PhaseVersionCreated
		}
		return *out, false, nil
	}
	out.Activated = true
	out.CanActivate = false
	out.Phase = PhaseActivated
	out.Activation = &activation.ActivateResult{
		ProcedureID: view.ProcedureID, ActiveVersionID: summary.ID,
		Version: summary.Version, Status: "ACTIVE", Noop: true,
	}
	if row.Status == repository.DraftStatusPublished {
		out.Reasons = []string{"DRAFT_ALREADY_PUBLISHED"}
		out.BlockedReason = ""
		out.Draft = *view
		return *out, true, nil
	}
	if pubErr := s.publishDraft(ctx, actor, draftID, row, view); pubErr != nil {
		out.ReconciliationRequired = true
		out.Phase = PhaseReconciliationRequired
		out.BlockedReason = "DRAFT_PUBLISH_PENDING"
		out.Reasons = []string{pubErr.Error()}
		out.Draft = *view
		return *out, true, nil
	}
	out.Draft = *view
	out.BlockedReason = ""
	out.Reasons = []string{"RECONCILED_PUBLISHED_AFTER_ACTIVE"}
	return *out, true, nil
}

func (s *Service) ensureVersion(ctx context.Context, actor, draftID uuid.UUID, req ActivateDraftRequest, row *repository.ProcedureDraft, view *DraftView, out *ActivateDraftResult) (uuid.UUID, error) {
	if view.PublishedVersionID != nil {
		out.Version = &activation.CreateVersionResult{
			ID: *view.PublishedVersionID, ProcedureID: view.ProcedureID,
			Version: view.PublishedVersion, Status: "APPROVED",
		}
		out.Phase = PhaseVersionCreated
		return *view.PublishedVersionID, nil
	}
	label := strings.TrimSpace(req.Version)
	if label == "" {
		label = stableVersionLabel(draftID)
	}
	created, err := s.Activation.CreateVersion(ctx, actor, draftOpID(draftID, "create-version"), view.ProcedureID, activation.CreateVersionRequest{
		Version:       label,
		Definition:    view.Definition,
		Confirm:       true,
		SourceDraftID: &draftID,
	})
	if err != nil {
		return uuid.Nil, err
	}
	out.Version = &created
	view.PublishedVersionID = &created.ID
	view.PublishedVersion = created.Version
	saved, persistErr := s.persistApprovedPayload(ctx, actor, draftID, row.UpdatedAt, *view)
	if persistErr != nil {
		reloaded, getErr := s.Drafts.Get(ctx, s.XAID, draftID)
		if getErr == nil {
			current, viewErr := viewFromRow(reloaded)
			if viewErr == nil && current.PublishedVersionID != nil && *current.PublishedVersionID == created.ID {
				*view = current
				*row = reloaded
				out.Draft = current
				out.Phase = PhaseVersionCreated
				return created.ID, nil
			}
		}
		if errors.Is(persistErr, repository.ErrNotFound) {
			return uuid.Nil, ErrConflict
		}
		return uuid.Nil, persistErr
	}
	if saved != nil {
		*view = *saved
		row.UpdatedAt = saved.UpdatedAt
	}
	out.Draft = *view
	out.Phase = PhaseVersionCreated
	return created.ID, nil
}

func (s *Service) metadataPersistFailure(ctx context.Context, out ActivateDraftResult, view DraftView, idx index.Result, persistErr error) (ActivateDraftResult, error) {
	out.Activated = false
	out.CanActivate = false
	out.BlockedReason = "INDEX_METADATA_PERSIST_FAILED"
	out.Reasons = []string{persistErr.Error()}
	out.Draft = persistedDraftView(ctx, s, view)
	if strings.EqualFold(idx.LinkStatus, "FAILED") || strings.EqualFold(idx.JobStatus, "FAILED") {
		out.Phase = PhaseIndexingFailed
		return out, nil
	}
	if strings.EqualFold(idx.LinkStatus, "READY") {
		out.Phase = PhaseSourceReady
		return out, nil
	}
	out.Phase = PhaseIndexingRunning
	out.IndexingPending = true
	return out, nil
}

// persistedDraftView returns the draft as stored in PostgreSQL. Unsaved
// IndexRequestID/IndexJobID mutations are never returned after a persist failure.
func persistedDraftView(ctx context.Context, s *Service, fallback DraftView) DraftView {
	cleared := fallback
	cleared.IndexRequestID = nil
	cleared.IndexJobID = nil
	if s == nil || s.Drafts == nil || fallback.ID == uuid.Nil {
		return cleared
	}
	row, err := s.Drafts.Get(ctx, s.XAID, fallback.ID)
	if err != nil {
		return cleared
	}
	current, viewErr := viewFromRow(row)
	if viewErr != nil {
		return cleared
	}
	return current
}

func (s *Service) persistApprovedPayload(ctx context.Context, actor, draftID uuid.UUID, expected time.Time, view DraftView) (*DraftView, error) {
	if s.Drafts == nil {
		return nil, fmt.Errorf("%w: draft store is not configured", ErrNotConfigured)
	}
	payload, err := json.Marshal(storedPayloadFromView(view))
	if err != nil {
		return nil, err
	}
	validationRaw, err := json.Marshal(view.Validation)
	if err != nil {
		return nil, err
	}
	saved, err := s.Drafts.UpdateStatus(ctx, s.XAID, draftID, actor, repository.DraftStatusApproved, payload, validationRaw, expected, []string{repository.DraftStatusApproved, repository.DraftStatusPublished})
	if err != nil {
		return nil, err
	}
	next, vErr := viewFromRow(saved)
	if vErr != nil {
		return nil, vErr
	}
	return &next, nil
}

func (s *Service) publishDraft(ctx context.Context, actor, draftID uuid.UUID, row repository.ProcedureDraft, view *DraftView) error {
	if s.Drafts == nil {
		return fmt.Errorf("%w: draft store is not configured", ErrNotConfigured)
	}
	pending := *view
	pending.Status = repository.DraftStatusPublished
	payload, err := json.Marshal(storedPayloadFromView(pending))
	if err != nil {
		return err
	}
	validationRaw, err := json.Marshal(pending.Validation)
	if err != nil {
		return err
	}
	saved, err := s.Drafts.UpdateStatus(ctx, s.XAID, draftID, actor, repository.DraftStatusPublished, payload, validationRaw, row.UpdatedAt, []string{repository.DraftStatusApproved, repository.DraftStatusPublished})
	if errors.Is(err, repository.ErrNotFound) {
		return fmt.Errorf("drafting: draft %s not found after activation (ACTIVE version preserved)", draftID)
	}
	if err != nil {
		return err
	}
	next, vErr := viewFromRow(saved)
	if vErr != nil {
		return vErr
	}
	*view = next
	return nil
}

func draftOpID(draftID uuid.UUID, op string) uuid.UUID {
	return uuid.NewSHA1(uuid.NameSpaceOID, []byte("cas.phase45.draft."+op+"."+draftID.String()))
}

func stableVersionLabel(draftID uuid.UUID) string {
	n := binary.BigEndian.Uint32(draftID[:4])%90000 + 10000
	m := binary.BigEndian.Uint16(draftID[4:6]) % 1000
	return fmt.Sprintf("8.%d.%d", n, m)
}

func sourceIndexState(summary activation.VersionSummary) string {
	for _, doc := range summary.Documents {
		if !strings.EqualFold(doc.RelationshipType, "SOURCE") {
			continue
		}
		status := strings.ToUpper(strings.TrimSpace(doc.IndexStatus))
		gen := strings.ToUpper(strings.TrimSpace(doc.GenerationStatus))
		if status == "READY" && (gen == "READY" || gen == "") {
			return "READY"
		}
		if status == "FAILED" || gen == "FAILED" {
			return "FAILED"
		}
		if status == "PROCESSING" || gen == "PROCESSING" {
			return "PROCESSING"
		}
	}
	return "UNKNOWN"
}

func pendingReasons(summary activation.VersionSummary) []string {
	if len(summary.Eligibility.Reasons) > 0 {
		return summary.Eligibility.Reasons
	}
	return []string{"DOCUMENT_INDEX_PROCESSING"}
}

func hasReason(reasons []string, code string) bool {
	for _, r := range reasons {
		if strings.EqualFold(r, code) {
			return true
		}
	}
	return false
}

func isIndexInProgress(err error) bool {
	if err == nil {
		return false
	}
	msg := err.Error()
	return strings.Contains(msg, "conflict") || strings.Contains(msg, "ErrConflict") || strings.Contains(msg, "invalid_state")
}
