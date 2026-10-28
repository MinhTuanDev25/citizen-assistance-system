package procedure

import (
	"errors"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/drafting"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/repository"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/response"
)

// PlanAIDraft previews how a long document would be drafted: the detected
// procedure sections, the selected one and its batches. Persists nothing.
func (h *Handler) PlanAIDraft(c *gin.Context) {
	if _, ok := h.adminActor(c); !ok {
		return
	}
	procedureID, _, ok := h.parseIDs(c)
	if !ok || !h.requireDrafting(c) {
		return
	}
	var body drafting.DraftPlanRequest
	if err := decodeJSON(c.Request.Body, &body); err != nil {
		response.Fail(c, http.StatusBadRequest, "VALIDATION", "invalid request body")
		return
	}
	plan, err := h.Drafting.PlanDraft(c.Request.Context(), procedureID, body)
	if err != nil {
		writeDraftJobErr(c, err)
		return
	}
	response.OK(c, plan)
}

// CreateAIDraftJob queues a multi-batch draft job (202). Replaying the same
// X-Request-ID with the same body returns the existing job (200).
func (h *Handler) CreateAIDraftJob(c *gin.Context) {
	actor, requestID, ok := h.adminRequest(c)
	if !ok {
		return
	}
	procedureID, _, ok := h.parseIDs(c)
	if !ok || !h.requireDrafting(c) {
		return
	}
	var body drafting.DraftPlanRequest
	if err := decodeJSON(c.Request.Body, &body); err != nil {
		response.Fail(c, http.StatusBadRequest, "VALIDATION", "invalid request body")
		return
	}
	job, created, err := h.Drafting.CreateDraftJob(c.Request.Context(), actor, requestID, procedureID, body)
	if err != nil {
		writeDraftJobErr(c, err)
		return
	}
	if created {
		response.Accepted(c, job)
		return
	}
	response.OK(c, job)
}

func (h *Handler) ListAIDraftJobs(c *gin.Context) {
	if _, ok := h.adminActor(c); !ok {
		return
	}
	procedureID, _, ok := h.parseIDs(c)
	if !ok || !h.requireDrafting(c) {
		return
	}
	items, err := h.Drafting.ListDraftJobs(c.Request.Context(), procedureID)
	if err != nil {
		writeDraftJobErr(c, err)
		return
	}
	response.OK(c, gin.H{"items": items, "count": len(items)})
}

func (h *Handler) GetAIDraftJob(c *gin.Context) {
	if _, ok := h.adminActor(c); !ok {
		return
	}
	jobID, ok := jobIDParam(c)
	if !ok || !h.requireDrafting(c) {
		return
	}
	job, err := h.Drafting.GetDraftJob(c.Request.Context(), jobID)
	if err != nil {
		writeDraftJobErr(c, err)
		return
	}
	response.OK(c, job)
}

// RetryAIDraftJob re-runs only the failed batches (or the merge) of a FAILED job.
func (h *Handler) RetryAIDraftJob(c *gin.Context) {
	if _, ok := h.adminActor(c); !ok {
		return
	}
	jobID, ok := jobIDParam(c)
	if !ok || !h.requireDrafting(c) {
		return
	}
	job, err := h.Drafting.RetryDraftJob(c.Request.Context(), jobID)
	if err != nil {
		writeDraftJobErr(c, err)
		return
	}
	response.Accepted(c, job)
}

// ResolveDraftConflict records the Admin's choice for one source conflict.
func (h *Handler) ResolveDraftConflict(c *gin.Context) {
	actor, ok := h.adminActor(c)
	if !ok || !h.requireDrafting(c) {
		return
	}
	draftID, err := uuid.Parse(c.Param("draftId"))
	if err != nil {
		response.Fail(c, http.StatusBadRequest, "VALIDATION", "invalid draft id")
		return
	}
	var body drafting.ResolveConflictRequest
	if err := decodeJSON(c.Request.Body, &body); err != nil {
		response.Fail(c, http.StatusBadRequest, "VALIDATION", "invalid request body")
		return
	}
	view, err := h.Drafting.ResolveConflict(c.Request.Context(), actor, draftID, body)
	if err != nil {
		writeDraftErr(c, err)
		return
	}
	response.OK(c, view)
}

// AcknowledgeDraftGap records why a coverage-gap field stays empty.
func (h *Handler) AcknowledgeDraftGap(c *gin.Context) {
	actor, ok := h.adminActor(c)
	if !ok || !h.requireDrafting(c) {
		return
	}
	draftID, err := uuid.Parse(c.Param("draftId"))
	if err != nil {
		response.Fail(c, http.StatusBadRequest, "VALIDATION", "invalid draft id")
		return
	}
	var body drafting.AcknowledgeGapRequest
	if err := decodeJSON(c.Request.Body, &body); err != nil {
		response.Fail(c, http.StatusBadRequest, "VALIDATION", "invalid request body")
		return
	}
	view, err := h.Drafting.AcknowledgeGap(c.Request.Context(), actor, draftID, body)
	if err != nil {
		writeDraftErr(c, err)
		return
	}
	response.OK(c, view)
}

// KeepDraftCitation puts a merge-dropped citation back into the Definition.
func (h *Handler) KeepDraftCitation(c *gin.Context) {
	actor, ok := h.adminActor(c)
	if !ok || !h.requireDrafting(c) {
		return
	}
	draftID, err := uuid.Parse(c.Param("draftId"))
	if err != nil {
		response.Fail(c, http.StatusBadRequest, "VALIDATION", "invalid draft id")
		return
	}
	var body drafting.KeepCitationRequest
	if err := decodeJSON(c.Request.Body, &body); err != nil {
		response.Fail(c, http.StatusBadRequest, "VALIDATION", "invalid request body")
		return
	}
	view, err := h.Drafting.KeepCitation(c.Request.Context(), actor, draftID, body)
	if err != nil {
		writeDraftErr(c, err)
		return
	}
	response.OK(c, view)
}

// DecideDraftWithoutSource keeps (or withdraws) a proposal without a source
// quotation on an Admin's stated reason.
func (h *Handler) DecideDraftWithoutSource(c *gin.Context) {
	actor, ok := h.adminActor(c)
	if !ok || !h.requireDrafting(c) {
		return
	}
	draftID, err := uuid.Parse(c.Param("draftId"))
	if err != nil {
		response.Fail(c, http.StatusBadRequest, "VALIDATION", "invalid draft id")
		return
	}
	var body drafting.AdminDecisionRequest
	if err := decodeJSON(c.Request.Body, &body); err != nil {
		response.Fail(c, http.StatusBadRequest, "VALIDATION", "invalid request body")
		return
	}
	view, err := h.Drafting.DecideWithoutSource(c.Request.Context(), actor, draftID, body)
	if err != nil {
		writeDraftErr(c, err)
		return
	}
	response.OK(c, view)
}

func jobIDParam(c *gin.Context) (uuid.UUID, bool) {
	id, err := uuid.Parse(c.Param("jobId"))
	if err != nil {
		response.Fail(c, http.StatusBadRequest, "VALIDATION", "invalid job id")
		return uuid.Nil, false
	}
	return id, true
}

func writeDraftJobErr(c *gin.Context, err error) {
	switch {
	case errors.Is(err, drafting.ErrPlanBlocked):
		reason := strings.TrimSpace(strings.TrimPrefix(err.Error(), drafting.ErrPlanBlocked.Error()+":"))
		response.Fail(c, http.StatusUnprocessableEntity, "PLAN_BLOCKED", reason)
	case errors.Is(err, drafting.ErrJobNotRetryable):
		response.Fail(c, http.StatusConflict, "JOB_NOT_RETRYABLE", "only a FAILED job can be retried")
	case errors.Is(err, repository.ErrIdempotency):
		response.Fail(c, http.StatusConflict, "IDEMPOTENCY_CONFLICT", "request id was reused with a different body")
	case strings.Contains(err.Error(), "base_draft_id"),
		strings.Contains(err.Error(), "invalid mode"),
		strings.Contains(err.Error(), "section_index"):
		response.Fail(c, http.StatusBadRequest, "VALIDATION", err.Error())
	default:
		writeDraftErr(c, err)
	}
}
