package response

import (
	"log/slog"
	"net/http"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/middleware"
	"github.com/gin-gonic/gin"
)

// Envelope is the standard JSON response shape.
type Envelope struct {
	RequestID string `json:"request_id"`
	Data      any    `json:"data,omitempty"`
	Error     *Error `json:"error,omitempty"`
}

type Error struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

func OK(c *gin.Context, data any) {
	c.JSON(http.StatusOK, Envelope{
		RequestID: middleware.GetRequestID(c),
		Data:      data,
	})
}

func Created(c *gin.Context, data any) {
	c.JSON(http.StatusCreated, Envelope{
		RequestID: middleware.GetRequestID(c),
		Data:      data,
	})
}

func Fail(c *gin.Context, status int, code, message string) {
	FailErr(c, status, code, message, nil)
}

// FailErr responds with error envelope and logs internal cause (for Loki).
func FailErr(c *gin.Context, status int, code, message string, err error) {
	log := middleware.LoggerFromContext(c, slog.Default())
	attrs := []any{
		"msg_type", "http_handler_error",
		"request_id", middleware.GetRequestID(c),
		"method", c.Request.Method,
		"path", c.Request.URL.Path,
		"status", status,
		"error_code", code,
		"error_message", message,
	}
	if err != nil {
		attrs = append(attrs, "error", err.Error())
		log.Error("handler_error", attrs...)
	} else if status >= 500 {
		log.Error("handler_error", attrs...)
	} else {
		log.Warn("handler_error", attrs...)
	}

	c.JSON(status, Envelope{
		RequestID: middleware.GetRequestID(c),
		Error: &Error{
			Code:    code,
			Message: message,
		},
	})
}
