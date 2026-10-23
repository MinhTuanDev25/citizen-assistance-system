package speechapi

import (
	"bytes"
	"context"
	"errors"
	"io"
	"log/slog"
	"mime"
	"net/http"
	"strings"
	"time"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/config"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/middleware"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/repository"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/response"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/speech"
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

const HeaderGuestToken = "X-Guest-Token"

// DefaultAuditTimeout is the bounded budget for a best-effort audit INSERT.
// The HTTP request context is not used: a client disconnect must not skip the write.
const DefaultAuditTimeout = 2 * time.Second

type SessionLoader interface {
	GetByID(ctx context.Context, id uuid.UUID) (*repository.Session, error)
}

type Auditor interface {
	Insert(ctx context.Context, rec speech.Record) error
}

type Handler struct {
	Enabled        bool
	Client         speech.Translator
	Sessions       SessionLoader
	Audit          Auditor
	MaxUploadBytes int64
	MaxBodyBytes   int64
	MaxInflight    int
	AuditTimeout   time.Duration
	inflight       chan struct{}
	Logger         *slog.Logger
}

func NewHandler(h Handler) *Handler {
	n := h.MaxInflight
	if n < 1 {
		n = 1
	}
	if h.inflight == nil || cap(h.inflight) != n {
		h.inflight = make(chan struct{}, n)
	}
	if h.AuditTimeout <= 0 {
		h.AuditTimeout = DefaultAuditTimeout
	}
	if h.Logger == nil {
		h.Logger = slog.Default()
	}
	return &h
}

type publicData struct {
	RequestID        string `json:"request_id"`
	ModelID          string `json:"model_id"`
	AsrTextBahnar    string `json:"asr_text_bahnar"`
	TranslatedTextVi string `json:"translated_text_vi"`
	DurationMS       int    `json:"duration_ms"`
	LatencyMS        int    `json:"latency_ms"`
}

// Translate godoc
//
//	@Summary		Translate Bahnar speech to Vietnamese
//	@Description	Authorizes the session from the URL, then forwards WAV to the internal C0 Speech API. Does not create a chat turn. Disabled unless SPEECH_ENABLED=true. Audit writes are best-effort: a successful translation may still return HTTP 200 if the audit INSERT fails.
//	@Tags			speech
//	@Accept			mpfd
//	@Produce		json
//	@Param			sessionId		path		string	true	"Conversation session UUID"
//	@Param			Authorization	header		string	false	"Bearer access token"
//	@Param			X-Guest-Token	header		string	false	"Guest token from session create"
//	@Param			X-Request-ID	header		string	false	"Correlation UUID"
//	@Param			audio			formData	file	true	"16 kHz mono WAV"
//	@Success		200				{object}	response.Envelope
//	@Failure		400				{object}	response.Envelope
//	@Failure		401				{object}	response.Envelope
//	@Failure		403				{object}	response.Envelope
//	@Failure		404				{object}	response.Envelope
//	@Failure		410				{object}	response.Envelope
//	@Failure		413				{object}	response.Envelope
//	@Failure		502				{object}	response.Envelope
//	@Failure		503				{object}	response.Envelope
//	@Failure		504				{object}	response.Envelope
//	@Router			/api/v1/sessions/{sessionId}/speech/translate [post]
func (h *Handler) Translate(c *gin.Context) {
	started := time.Now()
	reqID := middleware.GetRequestID(c)

	if !h.Enabled || h.Client == nil {
		response.Fail(c, http.StatusServiceUnavailable, "SPEECH_DISABLED", "speech translation is disabled")
		return
	}

	sessionID, parseErr := uuid.Parse(strings.TrimSpace(c.Param("sessionId")))
	if parseErr != nil {
		response.Fail(c, http.StatusBadRequest, "VALIDATION", "invalid sessionId")
		return
	}
	sess, ok := h.loadAuthorizedSession(c, sessionID)
	if !ok {
		return
	}
	if sess.Status != "OPEN" {
		response.Fail(c, http.StatusConflict, "CONFLICT", "session is not OPEN")
		return
	}

	select {
	case h.inflight <- struct{}{}:
		defer func() { <-h.inflight }()
	default:
		h.fail(c, speech.ErrOverloaded, sess, reqID, nil, 0)
		return
	}

	audio, filename, err := h.readAudio(c)
	if err != nil {
		h.fail(c, err, sess, reqID, nil, 0)
		return
	}

	result, inferErr := h.Client.Infer(c.Request.Context(), speech.InferRequest{
		Audio:     audio,
		Filename:  filename,
		RequestID: reqID,
	})
	audio = nil
	if inferErr != nil {
		h.fail(c, inferErr, sess, reqID, result, time.Since(started).Milliseconds())
		return
	}

	rid, _ := uuid.Parse(reqID)
	if rid == uuid.Nil {
		rid = uuid.New()
	}
	h.audit(c.Request.Context(), speech.Record{
		RequestID:            rid,
		SessionID:            sess.ID,
		XaID:                 sess.XaID,
		UserID:               sess.UserID,
		Outcome:              speech.OutcomeSuccess,
		HTTPStatus:           http.StatusOK,
		UpstreamStatus:       result.UpstreamStatus,
		DurationMS:           intPtr(result.DurationMS),
		LatencyMS:            intPtr(result.LatencyMS),
		ModelID:              result.ModelID,
		TokenizerParity:      result.TokenizerParity,
		ResearchProvenance:   result.ResearchProvenance,
		CheckpointIdentity:   result.CheckpointIdentity,
		GenerationMatchesRQ1: result.GenerationMatchesRQ1,
		ASRCheckpointStep:    result.ASRCheckpointStep,
		MTCheckpointStep:     result.MTCheckpointStep,
	})
	middleware.LoggerFromContext(c, h.Logger).Info("speech_translate",
		"request_id", reqID,
		"outcome", "SUCCESS",
		"status", http.StatusOK,
		"upstream_status", result.UpstreamStatus,
		"http_ms", time.Since(started).Milliseconds(),
		"duration_ms", result.DurationMS,
		"latency_ms", result.LatencyMS,
		"model_id", result.ModelID,
	)
	response.OK(c, publicData{
		RequestID:        reqID,
		ModelID:          result.ModelID,
		AsrTextBahnar:    result.AsrTextBahnar,
		TranslatedTextVi: result.TranslatedTextVi,
		DurationMS:       result.DurationMS,
		LatencyMS:        result.LatencyMS,
	})
}

// GoneLegacy rejects the removed POST /api/v1/speech/translate without reading the body.
func (h *Handler) GoneLegacy(c *gin.Context) {
	response.Fail(c, http.StatusGone, "DEPRECATED", "use POST /api/v1/sessions/:sessionId/speech/translate")
}

type publicErr struct {
	status  int
	code    string
	message string
}

func (h *Handler) fail(c *gin.Context, err error, sess *repository.Session, reqID string, result *speech.Result, httpMS int64) {
	mapped := mapPublic(err)
	if sess != nil {
		rid, _ := uuid.Parse(reqID)
		if rid == uuid.Nil {
			rid = uuid.New()
		}
		rec := speech.Record{
			RequestID:  rid,
			SessionID:  sess.ID,
			XaID:       sess.XaID,
			UserID:     sess.UserID,
			Outcome:    speech.OutcomeFailure,
			ErrorCode:  mapped.code,
			HTTPStatus: mapped.status,
		}
		if result != nil {
			rec.UpstreamStatus = result.UpstreamStatus
			rec.DurationMS = intPtr(result.DurationMS)
			rec.LatencyMS = intPtr(result.LatencyMS)
			rec.ModelID = result.ModelID
			rec.TokenizerParity = result.TokenizerParity
			rec.ResearchProvenance = result.ResearchProvenance
			rec.CheckpointIdentity = result.CheckpointIdentity
		}
		if rec.UpstreamStatus == 0 {
			rec.UpstreamStatus = speech.UpstreamStatus(err)
		}
		h.audit(c.Request.Context(), rec)
	}
	middleware.LoggerFromContext(c, h.Logger).Warn("speech_translate",
		"request_id", middleware.GetRequestID(c),
		"outcome", "FAILURE",
		"status", mapped.status,
		"error_code", mapped.code,
		"upstream_status", speech.UpstreamStatus(err),
		"upstream_code", speech.UpstreamCode(err),
		"http_ms", httpMS,
		"model_id", "c0_cascaded",
	)
	response.Fail(c, mapped.status, mapped.code, mapped.message)
}

// audit is best-effort. It uses a timeout detached from HTTP cancellation so a
// disconnect cannot skip the INSERT. Failures are logged and never change a
// successful citizen response to HTTP 500.
func (h *Handler) audit(reqCtx context.Context, rec speech.Record) {
	if h.Audit == nil {
		h.Logger.Error("speech_audit_failed",
			"error", "audit store is not configured",
			"request_id", rec.RequestID.String(),
			"outcome", rec.Outcome,
		)
		return
	}
	timeout := h.AuditTimeout
	if timeout <= 0 {
		timeout = DefaultAuditTimeout
	}
	parent := reqCtx
	if parent == nil {
		parent = context.Background()
	}
	ctx, cancel := context.WithTimeout(context.WithoutCancel(parent), timeout)
	defer cancel()
	if err := h.Audit.Insert(ctx, rec); err != nil {
		h.Logger.Error("speech_audit_failed",
			"error", err.Error(),
			"request_id", rec.RequestID.String(),
			"outcome", rec.Outcome,
			"http_status", rec.HTTPStatus,
		)
		return
	}
	h.Logger.Info("speech_audit_ok",
		"request_id", rec.RequestID.String(),
		"outcome", rec.Outcome,
		"http_status", rec.HTTPStatus,
	)
}

func (h *Handler) bodyLimit() int64 {
	return speech.EffectiveUpstreamBody(h.MaxBodyBytes)
}

func (h *Handler) readAudio(c *gin.Context) (audio []byte, filename string, err error) {
	max := h.MaxUploadBytes
	if max <= 0 {
		return nil, "", errTooLarge
	}
	if c.Request.ContentLength > max+middleware.SpeechMultipartOverhead {
		return nil, "", errTooLarge
	}
	mediaType, _, ctypeErr := mime.ParseMediaType(c.GetHeader("Content-Type"))
	if ctypeErr != nil || !strings.HasPrefix(mediaType, "multipart/") {
		return nil, "", errMalformed
	}
	reader, rerr := c.Request.MultipartReader()
	if rerr != nil {
		return nil, "", errMalformed
	}
	var sawAudio bool
	for {
		part, nerr := reader.NextPart()
		if errors.Is(nerr, io.EOF) {
			break
		}
		if nerr != nil {
			return nil, "", errMalformed
		}
		switch part.FormName() {
		case "audio":
			if sawAudio {
				_ = part.Close()
				return nil, "", errMalformed
			}
			sawAudio = true
			filename = part.FileName()
			buf, readErr := io.ReadAll(io.LimitReader(part, max+1))
			_ = part.Close()
			if readErr != nil {
				if isTooLarge(readErr) {
					return nil, "", errTooLarge
				}
				return nil, "", errMalformed
			}
			if int64(len(buf)) > max {
				return nil, "", errTooLarge
			}
			encoded, encErr := speech.EncodedMultipartLen(buf, filename)
			if encErr != nil || int64(encoded) > config.SpeechUploadHardMaxBytes {
				return nil, "", errTooLarge
			}
			if limit := h.bodyLimit(); limit > 0 && int64(encoded) > limit {
				return nil, "", errTooLarge
			}
			audio = buf
		default:
			_, _ = io.Copy(io.Discard, io.LimitReader(part, 1024))
			_ = part.Close()
		}
	}
	if !sawAudio {
		return nil, "", errMissingAudio
	}
	if len(audio) == 0 {
		return nil, "", errEmptyAudio
	}
	if !bytes.HasPrefix(audio, []byte("RIFF")) || len(audio) < 12 || !bytes.Equal(audio[8:12], []byte("WAVE")) {
		return nil, "", errUnsupportedAudio
	}
	return audio, filename, nil
}

func (h *Handler) loadAuthorizedSession(c *gin.Context, sessionID uuid.UUID) (*repository.Session, bool) {
	if h.Sessions == nil {
		response.Fail(c, http.StatusInternalServerError, "INTERNAL", "speech session lookup is not configured")
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

var (
	errTooLarge         = errors.New("speech: too large")
	errMalformed        = errors.New("speech: malformed multipart")
	errMissingAudio     = errors.New("speech: missing audio")
	errEmptyAudio       = errors.New("speech: empty audio")
	errUnsupportedAudio = errors.New("speech: unsupported audio")
)

func mapPublic(err error) publicErr {
	switch {
	case errors.Is(err, errTooLarge), errors.Is(err, speech.ErrTooLarge):
		return publicErr{http.StatusRequestEntityTooLarge, "PAYLOAD_TOO_LARGE", "request body exceeds size limit"}
	case errors.Is(err, errMissingAudio):
		return publicErr{http.StatusBadRequest, "VALIDATION", "audio field is required"}
	case errors.Is(err, errEmptyAudio):
		return publicErr{http.StatusBadRequest, "VALIDATION", "audio is empty"}
	case errors.Is(err, errUnsupportedAudio), errors.Is(err, speech.ErrInvalidAudio):
		return publicErr{http.StatusBadRequest, "VALIDATION", "audio is invalid or unsupported"}
	case errors.Is(err, errMalformed):
		return publicErr{http.StatusBadRequest, "VALIDATION", "invalid multipart upload"}
	case errors.Is(err, speech.ErrDisabled):
		return publicErr{http.StatusServiceUnavailable, "SPEECH_DISABLED", "speech translation is disabled"}
	case errors.Is(err, speech.ErrCanceled):
		return publicErr{http.StatusBadRequest, "SPEECH_CANCELLED", "request was cancelled"}
	case errors.Is(err, speech.ErrTimeout):
		return publicErr{http.StatusGatewayTimeout, "SPEECH_TIMEOUT", "speech inference timed out"}
	case errors.Is(err, speech.ErrOverloaded):
		return publicErr{http.StatusServiceUnavailable, "SPEECH_OVERLOADED", "speech inference is busy"}
	case errors.Is(err, speech.ErrNotReady):
		return publicErr{http.StatusServiceUnavailable, "SPEECH_NOT_READY", "speech service is not ready"}
	case errors.Is(err, speech.ErrShuttingDown):
		return publicErr{http.StatusServiceUnavailable, "SPEECH_UNAVAILABLE", "speech service is unavailable"}
	case errors.Is(err, speech.ErrUpstreamAuth):
		return publicErr{http.StatusBadGateway, "SPEECH_UPSTREAM_AUTH", "speech service authentication failed"}
	case errors.Is(err, speech.ErrInvalidResponse):
		return publicErr{http.StatusBadGateway, "SPEECH_INVALID_UPSTREAM", "speech service returned an invalid response"}
	case errors.Is(err, speech.ErrInferenceFailed), errors.Is(err, speech.ErrUnavailable):
		return publicErr{http.StatusBadGateway, "SPEECH_UNAVAILABLE", "speech service is unavailable"}
	default:
		return publicErr{http.StatusInternalServerError, "INTERNAL", "speech translation failed"}
	}
}

func isTooLarge(err error) bool {
	return err != nil && strings.Contains(strings.ToLower(err.Error()), "too large")
}

func intPtr(v int) *int { return &v }
