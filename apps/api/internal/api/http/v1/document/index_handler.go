package documentapi

import (
	"errors"
	"net/http"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/index"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/middleware"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/response"
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

// DocumentLinkRequest attaches one document to one procedure version.
type DocumentLinkRequest struct {
	ProcedureVersionID string `json:"procedure_version_id"`
	RelationshipType   string `json:"relationship_type"`
	PageRange          string `json:"page_range"`
}

// LinkTargets godoc
//
//	@Summary	List procedure versions that can receive a document
//	@Tags		admin-documents
//	@Produce	json
//	@Success	200	{object}	response.Envelope
//	@Failure	401	{object}	response.Envelope
//	@Failure	403	{object}	response.Envelope
//	@Failure	500	{object}	response.Envelope
//	@Security	BearerAuth
//	@Router		/api/v1/admin/documents/link-targets [get]
func (h *Handler) LinkTargets(c *gin.Context) {
	if h.Index == nil {
		c.Status(http.StatusNotFound)
		return
	}
	items, err := h.Index.ListTargets(c.Request.Context())
	if err != nil {
		response.Fail(c, http.StatusInternalServerError, "INTERNAL", "failed to list procedure versions")
		return
	}
	response.OK(c, gin.H{"items": items})
}

// ListLinks godoc
//
//	@Summary	List procedure versions attached to a document
//	@Description	Returns per-link index status. READY links include page_count, chunk_count, ocr_page_count, pipeline_version, and embedding_model_id from the active generation only. Staging generations are not returned.
//	@Tags		admin-documents
//	@Produce	json
//	@Param		id	path		string	true	"Document id"
//	@Success	200	{object}	response.Envelope
//	@Failure	401	{object}	response.Envelope
//	@Failure	403	{object}	response.Envelope
//	@Failure	404	{object}	response.Envelope
//	@Failure	500	{object}	response.Envelope
//	@Security	BearerAuth
//	@Router		/api/v1/admin/documents/{id}/links [get]
func (h *Handler) ListLinks(c *gin.Context) {
	if h.Index == nil {
		c.Status(http.StatusNotFound)
		return
	}
	id, err := uuid.Parse(c.Param("id"))
	if err != nil {
		response.Fail(c, http.StatusNotFound, "NOT_FOUND", "document not found")
		return
	}
	items, err := h.Index.ListLinks(c.Request.Context(), id)
	if err != nil {
		writeIndexErr(c, err)
		return
	}
	response.OK(c, gin.H{"items": items})
}

// Link godoc
//
//	@Summary	Attach a document to a procedure version
//	@Tags		admin-documents
//	@Accept		json
//	@Produce	json
//	@Param		id				path		string		true	"Document id"
//	@Param		X-Request-ID	header		string		true	"Idempotency key"
//	@Param		body			body		DocumentLinkRequest	true	"Link"
//	@Success	200				{object}	response.Envelope
//	@Success	201				{object}	response.Envelope
//	@Failure	400				{object}	response.Envelope
//	@Failure	401				{object}	response.Envelope
//	@Failure	403				{object}	response.Envelope
//	@Failure	404				{object}	response.Envelope
//	@Failure	409				{object}	response.Envelope
//	@Failure	422				{object}	response.Envelope
//	@Failure	500				{object}	response.Envelope
//	@Security	BearerAuth
//	@Router		/api/v1/admin/documents/{id}/links [post]
func (h *Handler) Link(c *gin.Context) {
	if h.Index == nil {
		c.Status(http.StatusNotFound)
		return
	}
	actor, requestID, ok := adminRequest(c)
	if !ok {
		return
	}
	id, err := uuid.Parse(c.Param("id"))
	if err != nil {
		response.Fail(c, http.StatusNotFound, "NOT_FOUND", "document not found")
		return
	}
	var body DocumentLinkRequest
	if err := index.DecodeStrict(c.Request.Body, &body); err != nil {
		response.Fail(c, http.StatusBadRequest, "BAD_REQUEST", "invalid request body")
		return
	}
	versionID, err := uuid.Parse(body.ProcedureVersionID)
	if err != nil {
		response.Fail(c, http.StatusUnprocessableEntity, "VALIDATION", "procedure version is invalid")
		return
	}
	link, replay, err := h.Index.Link(c.Request.Context(), index.LinkInput{
		DocumentID: id, ProcedureVersionID: versionID, RelationshipType: body.RelationshipType,
		PageRange: body.PageRange, ActorUserID: actor, RequestID: requestID,
	})
	if err != nil {
		writeIndexErr(c, err)
		return
	}
	if replay {
		response.OK(c, link)
		return
	}
	response.Created(c, link)
}

// Unlink godoc
//
//	@Summary	Detach a document from a procedure version
//	@Tags		admin-documents
//	@Produce	json
//	@Param		id				path	string	true	"Document id"
//	@Param		versionId		path	string	true	"Procedure version id"
//	@Param		X-Request-ID	header	string	true	"Idempotency key"
//	@Success	200				{object}	response.Envelope
//	@Failure	400				{object}	response.Envelope
//	@Failure	401				{object}	response.Envelope
//	@Failure	403				{object}	response.Envelope
//	@Failure	404				{object}	response.Envelope
//	@Failure	409				{object}	response.Envelope
//	@Failure	422				{object}	response.Envelope
//	@Failure	500				{object}	response.Envelope
//	@Security	BearerAuth
//	@Router		/api/v1/admin/documents/{id}/links/{versionId} [delete]
func (h *Handler) Unlink(c *gin.Context) {
	if h.Index == nil {
		c.Status(http.StatusNotFound)
		return
	}
	actor, requestID, ok := adminRequest(c)
	if !ok {
		return
	}
	id, err := uuid.Parse(c.Param("id"))
	if err != nil {
		response.Fail(c, http.StatusNotFound, "NOT_FOUND", "document not found")
		return
	}
	versionID, err := uuid.Parse(c.Param("versionId"))
	if err != nil {
		response.Fail(c, http.StatusNotFound, "NOT_FOUND", "document link not found")
		return
	}
	replay, err := h.Index.Unlink(c.Request.Context(), id, versionID, actor, requestID)
	if err != nil {
		writeIndexErr(c, err)
		return
	}
	response.OK(c, gin.H{"status": "unlinked", "idempotent_replay": replay})
}

// RequestIndex godoc
//
//	@Summary	Claim indexing for one document link
//	@Tags		admin-documents
//	@Produce	json
//	@Param		id				path	string	true	"Document id"
//	@Param		versionId		path	string	true	"Procedure version id"
//	@Param		X-Request-ID	header	string	true	"Idempotency key"
//	@Success	200				{object}	response.Envelope
//	@Failure	400				{object}	response.Envelope
//	@Failure	401				{object}	response.Envelope
//	@Failure	403				{object}	response.Envelope
//	@Failure	404				{object}	response.Envelope
//	@Failure	409				{object}	response.Envelope
//	@Failure	422				{object}	response.Envelope
//	@Failure	500				{object}	response.Envelope
//	@Security	BearerAuth
//	@Router		/api/v1/admin/documents/{id}/links/{versionId}/index [post]
func (h *Handler) RequestIndex(c *gin.Context) {
	h.runIndex(c, false, false)
}

// RetryIndex godoc
//
//	@Summary	Retry one link from FAILED or an expired PROCESSING lease
//	@Tags		admin-documents
//	@Produce	json
//	@Param		id				path	string	true	"Document id"
//	@Param		versionId		path	string	true	"Procedure version id"
//	@Param		X-Request-ID	header	string	true	"Idempotency key"
//	@Success	200				{object}	response.Envelope
//	@Failure	400				{object}	response.Envelope
//	@Failure	401				{object}	response.Envelope
//	@Failure	403				{object}	response.Envelope
//	@Failure	404				{object}	response.Envelope
//	@Failure	409				{object}	response.Envelope
//	@Failure	422				{object}	response.Envelope
//	@Failure	500				{object}	response.Envelope
//	@Security	BearerAuth
//	@Router		/api/v1/admin/documents/{id}/links/{versionId}/index/retry [post]
func (h *Handler) RetryIndex(c *gin.Context) {
	h.runIndex(c, true, false)
}

// Reindex godoc
//
//	@Summary	Reindex a READY link without dropping the active generation
//	@Tags		admin-documents
//	@Produce	json
//	@Param		id				path	string	true	"Document id"
//	@Param		versionId		path	string	true	"Procedure version id"
//	@Param		X-Request-ID	header	string	true	"Idempotency key"
//	@Success	200				{object}	response.Envelope
//	@Failure	400				{object}	response.Envelope
//	@Failure	401				{object}	response.Envelope
//	@Failure	403				{object}	response.Envelope
//	@Failure	404				{object}	response.Envelope
//	@Failure	409				{object}	response.Envelope
//	@Failure	422				{object}	response.Envelope
//	@Failure	500				{object}	response.Envelope
//	@Security	BearerAuth
//	@Router		/api/v1/admin/documents/{id}/links/{versionId}/index/reindex [post]
func (h *Handler) Reindex(c *gin.Context) {
	h.runIndex(c, false, true)
}

// IndexMetrics godoc
//
//	@Summary	Index generation counts for this commune
//	@Tags		admin-documents
//	@Produce	json
//	@Success	200	{object}	response.Envelope
//	@Failure	401	{object}	response.Envelope
//	@Failure	403	{object}	response.Envelope
//	@Security	BearerAuth
//	@Router		/api/v1/admin/documents/index-metrics [get]
func (h *Handler) IndexMetrics(c *gin.Context) {
	if h.Index == nil {
		c.Status(http.StatusNotFound)
		return
	}
	metrics, err := h.Index.Metrics(c.Request.Context())
	if err != nil {
		response.Fail(c, http.StatusInternalServerError, "INTERNAL", "failed to read index metrics")
		return
	}
	response.OK(c, metrics)
}

func (h *Handler) runIndex(c *gin.Context, retry bool, reindex bool) {
	if h.Index == nil {
		c.Status(http.StatusNotFound)
		return
	}
	actor, requestID, ok := adminRequest(c)
	if !ok {
		return
	}
	if c.Request.Body != nil && c.Request.ContentLength != 0 {
		var discard struct{}
		if err := index.DecodeStrict(c.Request.Body, &discard); err != nil {
			response.Fail(c, http.StatusBadRequest, "BAD_REQUEST", "invalid request body")
			return
		}
	}
	id, err := uuid.Parse(c.Param("id"))
	if err != nil {
		response.Fail(c, http.StatusNotFound, "NOT_FOUND", "document not found")
		return
	}
	versionID, err := uuid.Parse(c.Param("versionId"))
	if err != nil {
		response.Fail(c, http.StatusNotFound, "NOT_FOUND", "document link not found")
		return
	}
	result, err := h.Index.Request(c.Request.Context(), id, versionID, actor, requestID, retry)
	if reindex {
		result, err = h.Index.Reindex(c.Request.Context(), id, versionID, actor, requestID)
	}
	if err != nil {
		writeIndexErr(c, err)
		return
	}
	response.OK(c, result)
}

func adminRequest(c *gin.Context) (uuid.UUID, uuid.UUID, bool) {
	actor, ok := middleware.UserID(c)
	if !ok {
		response.Fail(c, http.StatusUnauthorized, "UNAUTHORIZED", "admin role required")
		return uuid.Nil, uuid.Nil, false
	}
	requestID, err := uuid.Parse(c.GetHeader("X-Request-ID"))
	if err != nil {
		response.Fail(c, http.StatusUnprocessableEntity, "VALIDATION", "X-Request-ID must be a UUID")
		return uuid.Nil, uuid.Nil, false
	}
	return actor, requestID, true
}

func writeIndexErr(c *gin.Context, err error) {
	switch {
	case errors.Is(err, index.ErrNotFound):
		response.Fail(c, http.StatusNotFound, "NOT_FOUND", "document not found")
	case errors.Is(err, index.ErrConflict):
		response.Fail(c, http.StatusConflict, "CONFLICT", "document indexing is already in progress")
	case errors.Is(err, index.ErrDuplicate):
		response.Fail(c, http.StatusConflict, "CONFLICT", "document is already attached")
	case errors.Is(err, index.ErrIdempotency):
		response.Fail(c, http.StatusConflict, "IDEMPOTENCY_CONFLICT", "request id was reused with a different body")
	case errors.Is(err, index.ErrInvalidState):
		response.Fail(c, http.StatusUnprocessableEntity, "VALIDATION", "document status does not allow this action")
	case errors.Is(err, index.ErrNotLinked):
		response.Fail(c, http.StatusUnprocessableEntity, "VALIDATION", "document is not attached to a procedure version")
	case errors.Is(err, index.ErrValidation):
		response.Fail(c, http.StatusUnprocessableEntity, "VALIDATION", "document and procedure version do not match")
	default:
		response.Fail(c, http.StatusInternalServerError, "INTERNAL", "indexing failed")
	}
}
