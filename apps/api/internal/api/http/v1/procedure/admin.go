package procedure

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/activation"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/middleware"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/response"
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

func (h *Handler) adminActor(c *gin.Context) (uuid.UUID, bool) {
	actor, ok := middleware.UserID(c)
	if !ok {
		response.Fail(c, http.StatusUnauthorized, "UNAUTHORIZED", "admin role required")
		return uuid.Nil, false
	}
	return actor, true
}

func (h *Handler) adminRequest(c *gin.Context) (uuid.UUID, uuid.UUID, bool) {
	actor, ok := h.adminActor(c)
	if !ok {
		return uuid.Nil, uuid.Nil, false
	}
	requestID, err := uuid.Parse(c.GetHeader("X-Request-ID"))
	if err != nil {
		response.Fail(c, http.StatusUnprocessableEntity, "VALIDATION", "X-Request-ID must be a UUID")
		return uuid.Nil, uuid.Nil, false
	}
	return actor, requestID, true
}

func (h *Handler) parseIDs(c *gin.Context) (uuid.UUID, uuid.UUID, bool) {
	procedureID, err := uuid.Parse(c.Param("procedureId"))
	if err != nil {
		response.Fail(c, http.StatusBadRequest, "VALIDATION", "invalid procedure id")
		return uuid.Nil, uuid.Nil, false
	}
	raw := c.Param("versionId")
	if raw == "" {
		return procedureID, uuid.Nil, true
	}
	versionID, err := uuid.Parse(raw)
	if err != nil {
		response.Fail(c, http.StatusBadRequest, "VALIDATION", "invalid version id")
		return uuid.Nil, uuid.Nil, false
	}
	return procedureID, versionID, true
}

// ListAdmin godoc
//
//	@Summary	List commune procedures including versions without ACTIVE
//	@Description	Admin-only. Server commune scope. Does not use ListActive. Marks seed ACTIVE rows without activated_at as legacy_active.
//	@Tags		admin-procedures
//	@Produce	json
//	@Success	200	{object}	response.Envelope
//	@Failure	401	{object}	response.Envelope
//	@Failure	403	{object}	response.Envelope
//	@Failure	500	{object}	response.Envelope
//	@Security	BearerAuth
//	@Router		/api/v1/admin/procedures [get]
func (h *Handler) ListAdmin(c *gin.Context) {
	if h.Activation == nil {
		c.Status(http.StatusNotFound)
		return
	}
	items, err := h.Activation.ListProcedures(c.Request.Context())
	if err != nil {
		writeActivationErr(c, err)
		return
	}
	response.OK(c, gin.H{"items": items, "count": len(items)})
}

// CreateProcedure godoc
//
//	@Summary	Create a procedure in the server commune
//	@Description	Admin-only. Body: procedure_code (^[a-z0-9_]{2,64}$, unique per commune), name, domain_id (active domain). Creates one SOURCE version 0.0.0 with an empty definition so the first PDF can be linked and indexed for an AI draft. Never creates an ACTIVE version. Idempotent per X-Request-ID; 409 PROCEDURE_CODE_EXISTS.
//	@Tags		admin-procedures
//	@Accept		json
//	@Produce	json
//	@Param		X-Request-ID	header		string	true	"Idempotency UUID"
//	@Success	201				{object}	response.Envelope
//	@Failure	400				{object}	response.Envelope
//	@Failure	401				{object}	response.Envelope
//	@Failure	403				{object}	response.Envelope
//	@Failure	409				{object}	response.Envelope
//	@Failure	422				{object}	response.Envelope
//	@Failure	500				{object}	response.Envelope
//	@Security	BearerAuth
//	@Router		/api/v1/admin/procedures [post]
func (h *Handler) CreateProcedure(c *gin.Context) {
	actor, requestID, ok := h.adminRequest(c)
	if !ok {
		return
	}
	if h.Activation == nil {
		c.Status(http.StatusNotFound)
		return
	}
	var body activation.CreateProcedureRequest
	if err := decodeJSON(c.Request.Body, &body); err != nil {
		response.Fail(c, http.StatusBadRequest, "VALIDATION", "invalid request body")
		return
	}
	out, err := h.Activation.CreateProcedure(c.Request.Context(), actor, requestID, body)
	if err != nil {
		writeActivationErr(c, err)
		return
	}
	if out.Replayed {
		response.OK(c, out)
		return
	}
	response.Created(c, out)
}

// ListVersions godoc
//
//	@Summary	List procedure versions with indexing eligibility
//	@Description	Admin-only. Includes APPROVED, INDEXING, ACTIVE, ARCHIVED. Eligibility requires published READY generations, not mock READY.
//	@Tags		admin-procedures
//	@Produce	json
//	@Param		procedureId	path		string	true	"Procedure UUID"
//	@Success	200			{object}	response.Envelope
//	@Failure	401			{object}	response.Envelope
//	@Failure	403			{object}	response.Envelope
//	@Failure	404			{object}	response.Envelope
//	@Failure	500			{object}	response.Envelope
//	@Security	BearerAuth
//	@Router		/api/v1/admin/procedures/{procedureId}/versions [get]
func (h *Handler) ListVersions(c *gin.Context) {
	if h.Activation == nil {
		c.Status(http.StatusNotFound)
		return
	}
	procedureID, _, ok := h.parseIDs(c)
	if !ok {
		return
	}
	items, err := h.Activation.ListVersions(c.Request.Context(), procedureID)
	if err != nil {
		writeActivationErr(c, err)
		return
	}
	response.OK(c, gin.H{"items": items, "count": len(items)})
}

// GetVersion godoc
//
//	@Summary	Get one procedure version, definition, and Activate eligibility
//	@Description	Admin-only. Reasons include MOCK_INDEX_NOT_PUBLISHED when a link is READY without a published generation.
//	@Tags		admin-procedures
//	@Produce	json
//	@Param		procedureId	path		string	true	"Procedure UUID"
//	@Param		versionId	path		string	true	"Version UUID"
//	@Success	200			{object}	response.Envelope
//	@Failure	401			{object}	response.Envelope
//	@Failure	403			{object}	response.Envelope
//	@Failure	404			{object}	response.Envelope
//	@Failure	500			{object}	response.Envelope
//	@Security	BearerAuth
//	@Router		/api/v1/admin/procedures/{procedureId}/versions/{versionId} [get]
func (h *Handler) GetVersion(c *gin.Context) {
	if h.Activation == nil {
		c.Status(http.StatusNotFound)
		return
	}
	procedureID, versionID, ok := h.parseIDs(c)
	if !ok {
		return
	}
	item, err := h.Activation.GetVersion(c.Request.Context(), procedureID, versionID)
	if err != nil {
		writeActivationErr(c, err)
		return
	}
	response.OK(c, item)
}

// CreateVersion godoc
//
//	@Summary	Create an APPROVED candidate procedure version
//	@Description	Admin-only. Copies definition from the current ACTIVE version unless definition is supplied. Requires confirm=true. Does not activate.
//	@Tags		admin-procedures
//	@Accept		json
//	@Produce	json
//	@Param		procedureId		path		string	true	"Procedure UUID"
//	@Param		X-Request-ID	header		string	true	"Idempotency UUID"
//	@Success	201				{object}	response.Envelope
//	@Failure	400				{object}	response.Envelope
//	@Failure	401				{object}	response.Envelope
//	@Failure	403				{object}	response.Envelope
//	@Failure	404				{object}	response.Envelope
//	@Failure	409				{object}	response.Envelope
//	@Failure	422				{object}	response.Envelope
//	@Failure	500				{object}	response.Envelope
//	@Security	BearerAuth
//	@Router		/api/v1/admin/procedures/{procedureId}/versions [post]
func (h *Handler) CreateVersion(c *gin.Context) {
	actor, requestID, ok := h.adminRequest(c)
	if !ok {
		return
	}
	procedureID, _, ok := h.parseIDs(c)
	if !ok {
		return
	}
	if h.Activation == nil {
		c.Status(http.StatusNotFound)
		return
	}
	var body activation.CreateVersionRequest
	if err := decodeJSON(c.Request.Body, &body); err != nil {
		response.Fail(c, http.StatusBadRequest, "VALIDATION", "invalid request body")
		return
	}
	out, err := h.Activation.CreateVersion(c.Request.Context(), actor, requestID, procedureID, body)
	if err != nil {
		writeActivationErr(c, err)
		return
	}
	if out.Replayed {
		response.OK(c, out)
		return
	}
	response.Created(c, out)
}

// Activate godoc
//
//	@Summary	Activate a procedure version atomically
//	@Description	Admin-only. Archives the current ACTIVE version, sets the candidate ACTIVE, updates procedures.active_version_id, writes PROCEDURE_VERSION_ACTIVATED. expected_active_version_id is required (null if none). 409 STALE_ACTIVE_VERSION / IDEMPOTENCY_CONFLICT / ACTIVATION_CONFLICT. 422 eligibility codes.
//	@Tags		admin-procedures
//	@Accept		json
//	@Produce	json
//	@Param		procedureId		path		string	true	"Procedure UUID"
//	@Param		versionId		path		string	true	"Version UUID"
//	@Param		X-Request-ID	header		string	true	"Idempotency UUID"
//	@Success	200				{object}	response.Envelope
//	@Failure	400				{object}	response.Envelope
//	@Failure	401				{object}	response.Envelope
//	@Failure	403				{object}	response.Envelope
//	@Failure	404				{object}	response.Envelope
//	@Failure	409				{object}	response.Envelope
//	@Failure	422				{object}	response.Envelope
//	@Failure	500				{object}	response.Envelope
//	@Security	BearerAuth
//	@Router		/api/v1/admin/procedures/{procedureId}/versions/{versionId}/activate [post]
func (h *Handler) Activate(c *gin.Context) {
	actor, requestID, ok := h.adminRequest(c)
	if !ok {
		return
	}
	procedureID, versionID, ok := h.parseIDs(c)
	if !ok {
		return
	}
	if h.Activation == nil {
		c.Status(http.StatusNotFound)
		return
	}
	expected, err := decodeExpected(c.Request.Body)
	if err != nil {
		response.Fail(c, http.StatusBadRequest, "VALIDATION", "expected_active_version_id is required")
		return
	}
	out, err := h.Activation.Activate(c.Request.Context(), actor, requestID, procedureID, versionID, expected)
	if err != nil {
		writeActivationErr(c, err)
		return
	}
	response.OK(c, out)
}

// ActivationHistory godoc
//
//	@Summary	Read-only activation audit for one procedure
//	@Description	Admin-only. Returns PROCEDURE_VERSION_ACTIVATED rows. Citizen/guest cannot call this route.
//	@Tags		admin-procedures
//	@Produce	json
//	@Param		procedureId	path		string	true	"Procedure UUID"
//	@Success	200			{object}	response.Envelope
//	@Failure	401			{object}	response.Envelope
//	@Failure	403			{object}	response.Envelope
//	@Failure	404			{object}	response.Envelope
//	@Failure	500			{object}	response.Envelope
//	@Security	BearerAuth
//	@Router		/api/v1/admin/procedures/{procedureId}/activation-history [get]
func (h *Handler) ActivationHistory(c *gin.Context) {
	if h.Activation == nil {
		c.Status(http.StatusNotFound)
		return
	}
	procedureID, _, ok := h.parseIDs(c)
	if !ok {
		return
	}
	items, err := h.Activation.History(c.Request.Context(), procedureID)
	if err != nil {
		writeActivationErr(c, err)
		return
	}
	response.OK(c, gin.H{"items": items})
}

func decodeJSON(r io.Reader, dest any) error {
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()
	return dec.Decode(dest)
}

func decodeExpected(r io.Reader) (*uuid.UUID, error) {
	var raw map[string]json.RawMessage
	if err := decodeJSON(r, &raw); err != nil {
		return nil, err
	}
	field, ok := raw["expected_active_version_id"]
	if !ok {
		return nil, errors.New("missing expected_active_version_id")
	}
	if string(field) == "null" {
		return nil, nil
	}
	var id uuid.UUID
	if err := json.Unmarshal(field, &id); err != nil {
		return nil, err
	}
	return &id, nil
}

func writeActivationErr(c *gin.Context, err error) {
	switch {
	case errors.Is(err, activation.ErrNotFound):
		response.Fail(c, http.StatusNotFound, "NOT_FOUND", "procedure version not found")
	case errors.Is(err, activation.ErrIdempotency):
		response.Fail(c, http.StatusConflict, "IDEMPOTENCY_CONFLICT", "request id was reused with a different body")
	case errors.Is(err, activation.ErrStale):
		response.Fail(c, http.StatusConflict, "STALE_ACTIVE_VERSION", "active version changed")
	case errors.Is(err, activation.ErrConflict):
		response.Fail(c, http.StatusConflict, "ACTIVATION_CONFLICT", "activation conflict")
	case errors.Is(err, activation.ErrDuplicateCode):
		response.Fail(c, http.StatusConflict, "PROCEDURE_CODE_EXISTS", "procedure_code already exists in this commune")
	case errors.Is(err, activation.ErrDuplicateVersion):
		response.Fail(c, http.StatusConflict, "ACTIVATION_CONFLICT", "version already exists")
	case errors.Is(err, activation.ErrNotApproved):
		response.Fail(c, http.StatusUnprocessableEntity, "VERSION_NOT_APPROVED", err.Error())
	case errors.Is(err, activation.ErrInvalidDefinition):
		response.Fail(c, http.StatusUnprocessableEntity, "INVALID_DEFINITION", err.Error())
	case errors.Is(err, activation.ErrNotEligible):
		response.Fail(c, http.StatusUnprocessableEntity, firstCode(err), err.Error())
	case errors.Is(err, activation.ErrValidation):
		response.Fail(c, http.StatusBadRequest, "VALIDATION", err.Error())
	default:
		response.FailErr(c, http.StatusInternalServerError, "INTERNAL", "activation failed", err)
	}
}

func firstCode(err error) string {
	msg := err.Error()
	for _, code := range []string{
		"MOCK_INDEX_NOT_PUBLISHED", "ACTIVE_GENERATION_MISSING", "DOCUMENT_INDEX_UPLOADED",
		"DOCUMENT_INDEX_PROCESSING", "DOCUMENT_INDEX_FAILED", "NO_SOURCE_DOCUMENT",
		"PARTIAL_SOURCE_INDEX", "DOCUMENT_NOT_CURRENT", "SOURCE_SCOPE_MISMATCH",
		"DOCUMENT_INDEX_NOT_READY", "DEFINITION_IDENTITY_MISMATCH",
	} {
		if strings.Contains(msg, code) {
			return code
		}
	}
	return "DOCUMENT_INDEX_NOT_READY"
}
