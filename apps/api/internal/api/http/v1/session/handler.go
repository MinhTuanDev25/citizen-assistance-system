package session

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/chat"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/middleware"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/repository"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/response"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/speech"
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

const HeaderGuestToken = "X-Guest-Token"

// SessionLookup is the session persistence used by Create / Turn / messages.
// *repository.SessionRepo implements it. The authorization rule is unchanged:
// JWT user_id or guest token must own the session. Turn does not compare xa_id
// against the deployment commune pin.
type SessionLookup interface {
	GetByID(ctx context.Context, id uuid.UUID) (*repository.Session, error)
	GetByGuestToken(ctx context.Context, guestToken string) (*repository.Session, error)
	CreateForUser(ctx context.Context, xaID string, userID uuid.UUID) (*repository.Session, error)
	CreateGuest(ctx context.Context, xaID, guestToken string) (*repository.Session, error)
	ListMessages(ctx context.Context, sessionID uuid.UUID, limit int) ([]repository.Message, error)
}

// TurnRunner is the chat turn dependency. *chat.Service implements it.
type TurnRunner interface {
	Turn(ctx context.Context, sessionID, requestID uuid.UUID, message string) (*chat.TurnResult, error)
}

// SpeechTurnLinker records chat_turn_request_id on a successful speech audit
// row. *speech.Store implements it. Optional: typed chat leaves it nil.
type SpeechTurnLinker interface {
	LinkChatTurn(ctx context.Context, speechRequestID, sessionID, chatTurnRequestID uuid.UUID) (speech.LinkResult, error)
}

type Handler struct {
	Sessions SessionLookup
	Communes *repository.CommuneRepo
	Turns    TurnRunner
	Speech   SpeechTurnLinker
	Logger   *slog.Logger
}

func NewHandler(sessions SessionLookup, communes *repository.CommuneRepo, turns TurnRunner) *Handler {
	return &Handler{Sessions: sessions, Communes: communes, Turns: turns}
}

type createRequest struct {
	XaID       string `json:"xa_id" binding:"required"`
	GuestToken string `json:"guest_token"`
}

type createMessageRequest struct {
	Message         string `json:"message" binding:"required"`
	SpeechRequestID string `json:"speech_request_id"`
}

// Create godoc
//
//	@Summary		Create conversation session
//	@Description	Guest: omit Authorization, optional guest_token resume. Logged-in: Bearer JWT → user_id set, no guest_token.
//	@Tags			sessions
//	@Accept			json
//	@Produce		json
//	@Param			Authorization	header		string			false	"Bearer access token"
//	@Param			body			body		createRequest	true	"xa_id required; guest_token optional (guest only)"
//	@Success		201				{object}	response.Envelope
//	@Failure		400				{object}	response.Envelope
//	@Failure		404				{object}	response.Envelope
//	@Failure		500				{object}	response.Envelope
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

	if userID, ok := middleware.UserID(c); ok {
		sess, err := h.Sessions.CreateForUser(c.Request.Context(), xaID, userID)
		if err != nil {
			response.FailErr(c, http.StatusInternalServerError, "INTERNAL", "failed to create session", err)
			return
		}
		response.Created(c, sess)
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
//	@Description	Returns the latest N messages ordered oldest→newest for UI replay.
//	@Tags			sessions
//	@Produce		json
//	@Param			sessionId		path		string	true	"Session UUID"
//	@Param			Authorization	header		string	false	"Bearer access token"
//	@Param			X-Guest-Token	header		string	false	"Guest token from session create"
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

// CreateMessage is deprecated. Use POST /sessions/:sessionId/turns.
//
//	@Summary		Deprecated — use POST /turns
//	@Description	Deprecated: use POST /api/v1/sessions/{sessionId}/turns instead. This route no longer accepts user messages.
//	@Tags			sessions
//	@Accept			json
//	@Produce		json
//	@Param			sessionId	path		string	true	"Session UUID"
//	@Failure		410			{object}	response.Envelope
//	@Router			/api/v1/sessions/{sessionId}/messages [post]
func (h *Handler) CreateMessage(c *gin.Context) {
	response.Fail(
		c,
		http.StatusGone,
		"DEPRECATED",
		"POST /sessions/:sessionId/messages is deprecated; use POST /sessions/:sessionId/turns",
	)
}

// Turn godoc
//
//	@Summary		Run one citizen chat turn
//	@Description	Atomically saves USER+ASSISTANT. Actions include ASK_MISSING_SLOTS, DIRECT_ANSWER, PROVIDE_FINAL_GUIDANCE, OUT_OF_SCOPE, CONFIRM_INTENT. Idempotent via X-Request-ID bound to normalized message hash; same ID + different body → 409 IDEMPOTENCY_CONFLICT. Optional speech_request_id links a prior SUCCESS speech audit row in the same session; linkage failures are logged and never change HTTP 200.
//	@Tags			sessions
//	@Accept			json
//	@Produce		json
//	@Param			sessionId		path		string					true	"Session UUID"
//	@Param			Authorization	header		string					false	"Bearer access token"
//	@Param			X-Guest-Token	header		string					false	"Guest token from session create"
//	@Param			X-Request-ID	header		string					true	"Idempotency key (UUID)"
//	@Param			body			body		createMessageRequest	true	"User message"
//	@Success		200				{object}	response.Envelope
//	@Failure		400				{object}	response.Envelope
//	@Failure		401				{object}	response.Envelope
//	@Failure		403				{object}	response.Envelope
//	@Failure		404				{object}	response.Envelope
//	@Failure		409				{object}	response.Envelope
//	@Failure		500				{object}	response.Envelope
//	@Router			/api/v1/sessions/{sessionId}/turns [post]
func (h *Handler) Turn(c *gin.Context) {
	sess, ok := h.loadAuthorizedSession(c)
	if !ok {
		return
	}
	if sess.Status != "OPEN" {
		response.Fail(c, http.StatusConflict, "CONFLICT", "session is not OPEN")
		return
	}
	if h.Turns == nil {
		response.Fail(c, http.StatusInternalServerError, "INTERNAL", "chat turn is not configured")
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
	if utf8.RuneCountInString(content) > chat.MaxMessageRunes {
		response.Fail(c, http.StatusBadRequest, "VALIDATION", "message is too long")
		return
	}

	rawRequestID := strings.TrimSpace(c.GetHeader(middleware.HeaderRequestID))
	if rawRequestID == "" {
		response.Fail(c, http.StatusBadRequest, "VALIDATION", "X-Request-ID header is required")
		return
	}
	requestID, err := uuid.Parse(rawRequestID)
	if err != nil {
		response.Fail(c, http.StatusBadRequest, "VALIDATION", "X-Request-ID must be a valid UUID")
		return
	}
	result, err := h.Turns.Turn(c.Request.Context(), sess.ID, requestID, content)
	if err != nil {
		switch {
		case errors.Is(err, repository.ErrNotFound):
			response.Fail(c, http.StatusNotFound, "NOT_FOUND", "session not found")
		case errors.Is(err, repository.ErrSessionNotOpen):
			response.Fail(c, http.StatusConflict, "CONFLICT", "session is not OPEN")
		case errors.Is(err, repository.ErrIdempotencyConflict):
			response.Fail(c, http.StatusConflict, "IDEMPOTENCY_CONFLICT", "X-Request-ID was reused with a different message payload")
		default:
			response.FailErr(c, http.StatusInternalServerError, "INTERNAL", "failed to run chat turn", err)
		}
		return
	}
	h.linkSpeechTurn(c, req.SpeechRequestID, sess.ID, requestID)
	if result != nil && result.Action != "" {
		c.Set("log_action", result.Action)
	}
	response.OK(c, result)
}

func (h *Handler) linkLogger(c *gin.Context) *slog.Logger {
	fallback := slog.Default()
	if h != nil && h.Logger != nil {
		fallback = h.Logger
	}
	return middleware.LoggerFromContext(c, fallback)
}

// linkSpeechTurn is best-effort. A missing or invalid speech_request_id, or a
// linker failure, must not change a successful Chat Turn into HTTP 500.
// Outcomes LINKED / ALREADY_LINKED / NOT_FOUND / CONFLICT / FAILED are logged
// with correlation UUIDs only — never audio, transcripts, or tokens.
func (h *Handler) linkSpeechTurn(c *gin.Context, rawSpeechID string, sessionID, chatTurnRequestID uuid.UUID) {
	if h == nil || h.Speech == nil {
		return
	}
	log := h.linkLogger(c)
	httpRequestID := middleware.GetRequestID(c)
	speechID, err := uuid.Parse(strings.TrimSpace(rawSpeechID))
	if err != nil || speechID == uuid.Nil {
		if strings.TrimSpace(rawSpeechID) != "" {
			log.Info("speech_chat_link",
				"outcome", "SKIPPED",
				"reason", "invalid_speech_request_id",
				"http_request_id", httpRequestID,
				"session_id", sessionID.String(),
				"chat_turn_request_id", chatTurnRequestID.String(),
			)
		}
		return
	}
	result, linkErr := h.Speech.LinkChatTurn(c.Request.Context(), speechID, sessionID, chatTurnRequestID)
	outcome := result.Outcome
	if outcome == "" {
		outcome = speech.LinkFailed
	}
	attrs := []any{
		"outcome", string(outcome),
		"speech_request_id", speechID.String(),
		"chat_turn_request_id", chatTurnRequestID.String(),
		"session_id", sessionID.String(),
		"http_request_id", httpRequestID,
	}
	if linkErr != nil {
		attrs = append(attrs, "error", linkErr.Error())
		log.Error("speech_chat_link", attrs...)
		return
	}
	switch outcome {
	case speech.LinkLinked, speech.LinkAlreadyLinked:
		log.Info("speech_chat_link", attrs...)
	default:
		log.Warn("speech_chat_link", attrs...)
	}
}

func (h *Handler) loadAuthorizedSession(c *gin.Context) (*repository.Session, bool) {
	sessionID, err := uuid.Parse(strings.TrimSpace(c.Param("sessionId")))
	if err != nil {
		response.Fail(c, http.StatusBadRequest, "VALIDATION", "invalid sessionId")
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

	if sess.UserID != nil {
		uid, ok := middleware.UserID(c)
		if !ok || uid != *sess.UserID {
			response.Fail(c, http.StatusForbidden, "FORBIDDEN", "session belongs to another user")
			return nil, false
		}
		return sess, true
	}

	guestToken := strings.TrimSpace(c.GetHeader(HeaderGuestToken))
	if guestToken == "" {
		response.Fail(c, http.StatusUnauthorized, "UNAUTHORIZED", "X-Guest-Token header is required")
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
		"id":                          s.ID,
		"xa_id":                       s.XaID,
		"status":                      s.Status,
		"active_procedure_id":         s.ActiveProcedureID,
		"active_procedure_version_id": s.ActiveProcedureVersionID,
		"created_at":                  s.CreatedAt,
		"updated_at":                  s.UpdatedAt,
	}
}
