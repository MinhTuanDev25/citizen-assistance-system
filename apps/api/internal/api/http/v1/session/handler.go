package session

import (
	"net/http"
	"strconv"
	"strings"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/middleware"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/repository"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/response"
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

const HeaderGuestToken = "X-Guest-Token"

type Handler struct {
	Sessions *repository.SessionRepo
	Communes *repository.CommuneRepo
}

func NewHandler(sessions *repository.SessionRepo, communes *repository.CommuneRepo) *Handler {
	return &Handler{Sessions: sessions, Communes: communes}
}

type createRequest struct {
	XaID       string `json:"xa_id" binding:"required"`
	GuestToken string `json:"guest_token"`
}

type createMessageRequest struct {
	Message string `json:"message" binding:"required"`
}

// Create godoc
//
//	@Summary		Create conversation session (guest)
//	@Tags			sessions
//	@Accept			json
//	@Produce		json
//	@Param			body	body		createRequest	true	"xa_id required; guest_token optional to resume"
//	@Success		201		{object}	response.Envelope
//	@Failure		400		{object}	response.Envelope
//	@Failure		404		{object}	response.Envelope
//	@Failure		500		{object}	response.Envelope
//	@Router			/api/v1/sessions [post]
func (h *Handler) Create(c *gin.Context) {
	var req createRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		response.Fail(c, http.StatusBadRequest, "VALIDATION", "xa_id is required")
		return
	}
	xaID := strings.TrimSpace(req.XaID)
	if xaID == "" {
		response.Fail(c, http.StatusBadRequest, "VALIDATION", "xa_id is required")
		return
	}

	commune, err := h.Communes.GetByID(c.Request.Context(), xaID)
	if err != nil {
		response.FailErr(c, http.StatusInternalServerError, "INTERNAL", "failed to validate commune", err)
		return
	}
	if commune == nil || !commune.IsActive {
		response.Fail(c, http.StatusNotFound, "NOT_FOUND", "commune not found or inactive")
		return
	}

	guestToken := strings.TrimSpace(req.GuestToken)
	if guestToken != "" {
		existing, err := h.Sessions.GetByGuestToken(c.Request.Context(), guestToken)
		if err != nil {
			response.FailErr(c, http.StatusInternalServerError, "INTERNAL", "failed to lookup session", err)
			return
		}
		if existing != nil {
			if existing.Status != "OPEN" {
				response.Fail(c, http.StatusConflict, "CONFLICT", "guest session is not OPEN; omit guest_token to create a new one")
				return
			}
			if existing.XaID != xaID {
				response.Fail(c, http.StatusConflict, "CONFLICT", "guest_token is bound to another xa_id")
				return
			}
			response.Created(c, existing)
			return
		}
	} else {
		guestToken = uuid.NewString()
	}

	sess, err := h.Sessions.CreateGuest(c.Request.Context(), xaID, guestToken)
	if err != nil {
		response.FailErr(c, http.StatusInternalServerError, "INTERNAL", "failed to create session", err)
		return
	}
	response.Created(c, sess)
}

// ListMessages godoc
//
//	@Summary		List messages for chat UI
//	@Description	Oldest→newest. FE maps role USER→user bubble, ASSISTANT→bot bubble.
//	@Tags			sessions
//	@Produce		json
//	@Param			sessionId		path		string	true	"Session UUID"
//	@Param			X-Guest-Token	header		string	true	"Guest token from session create"
//	@Param			limit			query		int		false	"Max messages (default 100, max 200)"
//	@Success		200				{object}	response.Envelope
//	@Failure		400				{object}	response.Envelope
//	@Failure		401				{object}	response.Envelope
//	@Failure		403				{object}	response.Envelope
//	@Failure		404				{object}	response.Envelope
//	@Failure		500				{object}	response.Envelope
//	@Router			/api/v1/sessions/{sessionId}/messages [get]
func (h *Handler) ListMessages(c *gin.Context) {
	sess, ok := h.loadAuthorizedSession(c)
	if !ok {
		return
	}

	limit := 100
	if raw := strings.TrimSpace(c.Query("limit")); raw != "" {
		n, err := strconv.Atoi(raw)
		if err != nil || n < 1 {
			response.Fail(c, http.StatusBadRequest, "VALIDATION", "limit must be a positive integer")
			return
		}
		limit = n
	}

	items, err := h.Sessions.ListMessages(c.Request.Context(), sess.ID, limit)
	if err != nil {
		response.FailErr(c, http.StatusInternalServerError, "INTERNAL", "failed to list messages", err)
		return
	}

	response.OK(c, gin.H{
		"session": sessionSummary(sess),
		"items":   items,
		"count":   len(items),
	})
}

// CreateMessage godoc
//
//	@Summary		Append USER message to session
//	@Description	Persists user message. Chat turn / Decision Engine will be added next.
//	@Tags			sessions
//	@Accept			json
//	@Produce		json
//	@Param			sessionId		path		string					true	"Session UUID"
//	@Param			X-Guest-Token	header		string					true	"Guest token from session create"
//	@Param			body			body		createMessageRequest	true	"User message"
//	@Success		201				{object}	response.Envelope
//	@Failure		400				{object}	response.Envelope
//	@Failure		401				{object}	response.Envelope
//	@Failure		403				{object}	response.Envelope
//	@Failure		404				{object}	response.Envelope
//	@Failure		409				{object}	response.Envelope
//	@Failure		500				{object}	response.Envelope
//	@Router			/api/v1/sessions/{sessionId}/messages [post]
func (h *Handler) CreateMessage(c *gin.Context) {
	sess, ok := h.loadAuthorizedSession(c)
	if !ok {
		return
	}
	if sess.Status != "OPEN" {
		response.Fail(c, http.StatusConflict, "CONFLICT", "session is not OPEN")
		return
	}

	var req createMessageRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		response.Fail(c, http.StatusBadRequest, "VALIDATION", "message is required")
		return
	}
	content := strings.TrimSpace(req.Message)
	if content == "" {
		response.Fail(c, http.StatusBadRequest, "VALIDATION", "message is required")
		return
	}

	requestID := parseOrNewRequestID(middleware.GetRequestID(c))

	msg, err := h.Sessions.InsertUserMessage(c.Request.Context(), sess.ID, requestID, content)
	if err != nil {
		response.FailErr(c, http.StatusInternalServerError, "INTERNAL", "failed to save message", err)
		return
	}
	_ = h.Sessions.TouchUpdatedAt(c.Request.Context(), sess.ID)

	response.Created(c, gin.H{
		"session_id": sess.ID.String(),
		"message":    msg,
		"note":       "user message saved; chat turn / Decision Engine not wired yet",
	})
}

func (h *Handler) loadAuthorizedSession(c *gin.Context) (*repository.Session, bool) {
	sessionID, err := uuid.Parse(strings.TrimSpace(c.Param("sessionId")))
	if err != nil {
		response.Fail(c, http.StatusBadRequest, "VALIDATION", "invalid sessionId")
		return nil, false
	}

	guestToken := strings.TrimSpace(c.GetHeader(HeaderGuestToken))
	if guestToken == "" {
		response.Fail(c, http.StatusUnauthorized, "UNAUTHORIZED", "X-Guest-Token header is required")
		return nil, false
	}

	sess, err := h.Sessions.GetByID(c.Request.Context(), sessionID)
	if err != nil {
		response.FailErr(c, http.StatusInternalServerError, "INTERNAL", "failed to load session", err)
		return nil, false
	}
	if sess == nil {
		response.Fail(c, http.StatusNotFound, "NOT_FOUND", "session not found")
		return nil, false
	}
	if sess.GuestToken == nil || *sess.GuestToken != guestToken {
		response.Fail(c, http.StatusForbidden, "FORBIDDEN", "guest token mismatch")
		return nil, false
	}
	return sess, true
}

func sessionSummary(s *repository.Session) gin.H {
	return gin.H{
		"id":                           s.ID,
		"xa_id":                        s.XaID,
		"status":                       s.Status,
		"active_procedure_id":          s.ActiveProcedureID,
		"active_procedure_version_id":  s.ActiveProcedureVersionID,
		"created_at":                   s.CreatedAt,
		"updated_at":                   s.UpdatedAt,
	}
}

func parseOrNewRequestID(raw string) uuid.UUID {
	if id, err := uuid.Parse(strings.TrimSpace(raw)); err == nil {
		return id
	}
	return uuid.New()
}
