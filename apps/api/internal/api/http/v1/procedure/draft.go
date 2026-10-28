package procedure

import (
	"errors"
	"net/http"
	"strings"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/activation"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/drafting"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/repository"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/response"
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

func (h *Handler) requireDrafting(c *gin.Context) bool {
	if h.Drafting != nil {
		return true
	}
	response.Fail(c, http.StatusServiceUnavailable, "DRAFTING_DISABLED", "definition drafting is not configured")
	return false
}

// GenerateAIDraft is the single-call draft for a small, explicit page range
// (source_page_start/end required). The admin UI drafts long documents through
// draft jobs (CreateAIDraftJob) instead.
func (h *Handler) GenerateAIDraft(c *gin.Context) {
	actor, requestID, ok := h.adminRequest(c)
	if !ok {
		return
	}
	procedureID, _, ok := h.parseIDs(c)
	if !ok {
		return
	}
	if !h.requireDrafting(c) {
		return
	}
	var body drafting.GenerateRequest
	if err := decodeJSON(c.Request.Body, &body); err != nil {
		response.Fail(c, http.StatusBadRequest, "VALIDATION", "invalid request body")
		return
	}
	view, err := h.Drafting.Generate(c.Request.Context(), actor, requestID, procedureID, body)
	if err != nil {
		writeDraftErr(c, err)
		return
	}
	response.Created(c, view)
}

func (h *Handler) ListDrafts(c *gin.Context) {
	if _, ok := h.adminActor(c); !ok {
		return
	}
	procedureID, _, ok := h.parseIDs(c)
	if !ok {
		return
	}
	if !h.requireDrafting(c) {
		return
	}
	items, err := h.Drafting.List(c.Request.Context(), procedureID)
	if err != nil {
		writeDraftErr(c, err)
		return
	}
	response.OK(c, gin.H{"items": items, "count": len(items)})
}

func (h *Handler) GetDraft(c *gin.Context) {
	if _, ok := h.adminActor(c); !ok {
		return
	}
	if !h.requireDrafting(c) {
		return
	}
	draftID, err := uuid.Parse(c.Param("draftId"))
	if err != nil {
		response.Fail(c, http.StatusBadRequest, "VALIDATION", "invalid draft id")
		return
	}
	view, err := h.Drafting.Get(c.Request.Context(), draftID)
	if err != nil {
		writeDraftErr(c, err)
		return
	}
	response.OK(c, view)
}

func (h *Handler) ValidateDraft(c *gin.Context) {
	if _, ok := h.adminActor(c); !ok {
		return
	}
	if !h.requireDrafting(c) {
		return
	}
	draftID, err := uuid.Parse(c.Param("draftId"))
	if err != nil {
		response.Fail(c, http.StatusBadRequest, "VALIDATION", "invalid draft id")
		return
	}
	view, err := h.Drafting.Validate(c.Request.Context(), draftID)
	if err != nil {
		writeDraftErr(c, err)
		return
	}
	response.OK(c, view)
}

func (h *Handler) PatchDraft(c *gin.Context) {
	actor, ok := h.adminActor(c)
	if !ok {
		return
	}
	if !h.requireDrafting(c) {
		return
	}
	draftID, err := uuid.Parse(c.Param("draftId"))
	if err != nil {
		response.Fail(c, http.StatusBadRequest, "VALIDATION", "invalid draft id")
		return
	}
	var body drafting.PatchRequest
	if err := decodeJSON(c.Request.Body, &body); err != nil {
		response.Fail(c, http.StatusBadRequest, "VALIDATION", "invalid request body")
		return
	}
	view, err := h.Drafting.Patch(c.Request.Context(), actor, draftID, body)
	if err != nil {
		writeDraftErr(c, err)
		return
	}
	response.OK(c, view)
}

// CloneDraft godoc
// @Summary     Copy an approved/published draft into a new editable draft
// @Tags        admin-drafts
// @Produce     json
// @Param       draftId path string true "Draft ID"
// @Success     201 {object} response.Envelope
// @Router      /admin/procedure-drafts/{draftId}/clone [post]
func (h *Handler) CloneDraft(c *gin.Context) {
	actor, ok := h.adminActor(c)
	if !ok {
		return
	}
	if !h.requireDrafting(c) {
		return
	}
	draftID, err := uuid.Parse(c.Param("draftId"))
	if err != nil {
		response.Fail(c, http.StatusBadRequest, "VALIDATION", "invalid draft id")
		return
	}
	view, err := h.Drafting.CloneDraft(c.Request.Context(), actor, draftID)
	if err != nil {
		writeDraftErr(c, err)
		return
	}
	response.Created(c, view)
}

func (h *Handler) ListAllDrafts(c *gin.Context) {
	if _, ok := h.adminActor(c); !ok {
		return
	}
	if !h.requireDrafting(c) {
		return
	}
	var procedureID *uuid.UUID
	if raw := strings.TrimSpace(c.Query("procedure_id")); raw != "" {
		id, err := uuid.Parse(raw)
		if err != nil {
			response.Fail(c, http.StatusBadRequest, "VALIDATION", "invalid procedure_id")
			return
		}
		procedureID = &id
	}
	items, err := h.Drafting.ListAll(c.Request.Context(), procedureID)
	if err != nil {
		writeDraftErr(c, err)
		return
	}
	response.OK(c, gin.H{"items": items, "count": len(items)})
}

func (h *Handler) ListDraftSources(c *gin.Context) {
	if _, ok := h.adminActor(c); !ok {
		return
	}
	procedureID, _, ok := h.parseIDs(c)
	if !ok {
		return
	}
	if !h.requireDrafting(c) {
		return
	}
	items, err := h.Drafting.ListSources(c.Request.Context(), procedureID)
	if err != nil {
		writeDraftErr(c, err)
		return
	}
	response.OK(c, gin.H{"items": items, "count": len(items)})
}

func (h *Handler) ReviewDraft(c *gin.Context) {
	actor, ok := h.adminActor(c)
	if !ok {
		return
	}
	if !h.requireDrafting(c) {
		return
	}
	draftID, err := uuid.Parse(c.Param("draftId"))
	if err != nil {
		response.Fail(c, http.StatusBadRequest, "VALIDATION", "invalid draft id")
		return
	}
	var body drafting.ReviewRequest
	if err := decodeJSON(c.Request.Body, &body); err != nil {
		response.Fail(c, http.StatusBadRequest, "VALIDATION", "invalid request body")
		return
	}
	view, err := h.Drafting.Review(c.Request.Context(), actor, draftID, body)
	if err != nil {
		writeDraftErr(c, err)
		return
	}
	response.OK(c, view)
}

func (h *Handler) ApproveDraft(c *gin.Context) {
	actor, ok := h.adminActor(c)
	if !ok {
		return
	}
	if !h.requireDrafting(c) {
		return
	}
	draftID, err := uuid.Parse(c.Param("draftId"))
	if err != nil {
		response.Fail(c, http.StatusBadRequest, "VALIDATION", "invalid draft id")
		return
	}
	var body drafting.ApproveRequest
	if err := decodeJSON(c.Request.Body, &body); err != nil {
		response.Fail(c, http.StatusBadRequest, "VALIDATION", "invalid request body")
		return
	}
	view, err := h.Drafting.Approve(c.Request.Context(), actor, draftID, body)
	if err != nil {
		writeDraftErr(c, err)
		return
	}
	response.OK(c, view)
}

func (h *Handler) ActivateDraft(c *gin.Context) {
	actor, requestID, ok := h.adminRequest(c)
	if !ok {
		return
	}
	if !h.requireDrafting(c) {
		return
	}
	draftID, err := uuid.Parse(c.Param("draftId"))
	if err != nil {
		response.Fail(c, http.StatusBadRequest, "VALIDATION", "invalid draft id")
		return
	}
	var body drafting.ActivateDraftRequest
	if err := decodeJSON(c.Request.Body, &body); err != nil {
		response.Fail(c, http.StatusBadRequest, "VALIDATION", "invalid request body")
		return
	}
	result, err := h.Drafting.Activate(c.Request.Context(), actor, requestID, draftID, body)
	if err != nil {
		writeDraftErr(c, err)
		return
	}
	response.OK(c, result)
}

func writeDraftErr(c *gin.Context, err error) {
	switch {
	case errors.Is(err, repository.ErrNotFound), errors.Is(err, drafting.ErrSourceNotFound):
		response.Fail(c, http.StatusNotFound, "NOT_FOUND", "draft or source not found")
	case errors.Is(err, drafting.ErrSourceTooLarge):
		response.Fail(c, http.StatusUnprocessableEntity, "SOURCE_TOO_LARGE", err.Error())
	case errors.Is(err, drafting.ErrUnverifiedSource):
		response.Fail(c, http.StatusUnprocessableEntity, "UNVERIFIED_SOURCE", "selected source is not a READY verified generation")
	case errors.Is(err, drafting.ErrNotConfigured), errors.Is(err, drafting.ErrDisabled):
		response.Fail(c, http.StatusServiceUnavailable, "DRAFTING_DISABLED", "definition drafting is not configured")
	case errors.Is(err, drafting.ErrDraftTimeout):
		response.Fail(c, http.StatusGatewayTimeout, "DRAFT_TIMEOUT", "ai draft timed out")
	case errors.Is(err, drafting.ErrRateLimited):
		response.Fail(c, http.StatusTooManyRequests, "RATE_LIMITED", "ai service rate limited")
	case errors.Is(err, drafting.ErrConflict):
		response.Fail(c, http.StatusConflict, "REVISION_CONFLICT", "draft revision is stale")
	case errors.Is(err, drafting.ErrApprovalBlocked):
		response.Fail(c, http.StatusUnprocessableEntity, "APPROVAL_BLOCKED", err.Error())
	case errors.Is(err, drafting.ErrReviewInvalid):
		response.Fail(c, http.StatusUnprocessableEntity, "REVIEW_INVALID", err.Error())
	case errors.Is(err, activation.ErrStale):
		response.Fail(c, http.StatusConflict, "STALE_ACTIVE_VERSION", "expected_active_version_id does not match the current ACTIVE version")
	case errors.Is(err, drafting.ErrNotApproved):
		response.Fail(c, http.StatusUnprocessableEntity, "NOT_APPROVED", "draft must be Admin-approved before activation")
	case strings.Contains(err.Error(), "document_id is required"),
		strings.Contains(err.Error(), "expected_updated_at"),
		strings.Contains(err.Error(), "confirm is required"),
		strings.Contains(err.Error(), "invalid page"),
		strings.Contains(err.Error(), "invalid review"):
		response.Fail(c, http.StatusBadRequest, "VALIDATION", err.Error())
	case strings.Contains(err.Error(), "unexpected status"),
		strings.Contains(err.Error(), "request failed"),
		errors.Is(err, drafting.ErrInvalidDraft):
		response.Fail(c, http.StatusBadGateway, "AI_PROVIDER", "draft generation failed")
	default:
		response.FailErr(c, http.StatusInternalServerError, "INTERNAL", "draft workflow failed", err)
	}
}
