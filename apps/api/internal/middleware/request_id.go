package middleware

import (
	"bytes"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

const (
	HeaderRequestID  = "X-Request-ID"
	ContextRequestID = "request_id"
	ContextLogger    = "logger"

	maxBodyLogBytes = 4096
)

// RequestID ensures every request has a UUID request_id (header + Gin context).
func RequestID() gin.HandlerFunc {
	return func(c *gin.Context) {
		rid := c.GetHeader(HeaderRequestID)
		if rid == "" {
			rid = uuid.NewString()
		}
		c.Set(ContextRequestID, rid)
		c.Writer.Header().Set(HeaderRequestID, rid)
		c.Next()
	}
}

// GetRequestID returns request_id from Gin context, or empty string.
func GetRequestID(c *gin.Context) string {
	if v, ok := c.Get(ContextRequestID); ok {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return ""
}

// LoggerFromContext returns request-scoped logger, or fallback.
func LoggerFromContext(c *gin.Context, fallback *slog.Logger) *slog.Logger {
	if v, ok := c.Get(ContextLogger); ok {
		if l, ok := v.(*slog.Logger); ok && l != nil {
			return l
		}
	}
	return fallback
}

type bodyLogWriter struct {
	gin.ResponseWriter
	body *bytes.Buffer
}

func (w *bodyLogWriter) Write(b []byte) (int, error) {
	if w.body.Len() < maxBodyLogBytes {
		remain := maxBodyLogBytes - w.body.Len()
		if len(b) > remain {
			_, _ = w.body.Write(b[:remain])
		} else {
			_, _ = w.body.Write(b)
		}
	}
	return w.ResponseWriter.Write(b)
}

// ApiLog writes one full JSON access line per request (stdout → Promtail/Loki).
func ApiLog(logger *slog.Logger) gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		rid := GetRequestID(c)
		reqLog := logger.With("request_id", rid)
		c.Set(ContextLogger, reqLog)

		reqBody := readRequestBody(c)

		blw := &bodyLogWriter{ResponseWriter: c.Writer, body: &bytes.Buffer{}}
		c.Writer = blw

		c.Next()

		status := c.Writer.Status()
		latency := time.Since(start).Milliseconds()
		attrs := []any{
			"msg_type", "http_access",
			"request_id", rid,
			"method", c.Request.Method,
			"path", c.Request.URL.Path,
			"query", c.Request.URL.RawQuery,
			"status", status,
			"latency_ms", latency,
			"client_ip", c.ClientIP(),
			"user_agent", c.Request.UserAgent(),
			"request_body", truncateBody(redactJSON(reqBody)),
			"response_body", truncateBody(blw.body.String()),
			"bytes_out", c.Writer.Size(),
		}

		switch {
		case status >= 500:
			reqLog.Error("http_request", attrs...)
		case status >= 400:
			reqLog.Warn("http_request", attrs...)
		default:
			// Keep /health|/ready quieter unless failure (still Info for Loki completeness at debug)
			if isProbe(c.Request.URL.Path) {
				reqLog.Debug("http_request", attrs...)
			} else {
				reqLog.Info("http_request", attrs...)
			}
		}
	}
}

func isProbe(path string) bool {
	return path == "/health" || path == "/ready" || strings.HasPrefix(path, "/swagger")
}

func readRequestBody(c *gin.Context) string {
	if c.Request.Body == nil || c.Request.Method == http.MethodGet || c.Request.Method == http.MethodHead || c.Request.Method == http.MethodDelete {
		return ""
	}
	raw, err := io.ReadAll(io.LimitReader(c.Request.Body, maxBodyLogBytes+1))
	if err != nil {
		return ""
	}
	_ = c.Request.Body.Close()
	c.Request.Body = io.NopCloser(bytes.NewBuffer(raw))
	return string(raw)
}

func truncateBody(s string) string {
	if s == "" {
		return ""
	}
	if !utf8.ValidString(s) {
		return "[binary]"
	}
	if len(s) > maxBodyLogBytes {
		return s[:maxBodyLogBytes] + "...[truncated]"
	}
	return s
}

// redactJSON does light masking for common secret field names (best-effort).
func redactJSON(s string) string {
	if s == "" {
		return s
	}
	lower := strings.ToLower(s)
	keys := []string{`"password"`, `"password_hash"`, `"token"`, `"access_token"`, `"refresh_token"`, `"secret"`, `"jwt"`}
	out := s
	for _, k := range keys {
		if !strings.Contains(lower, strings.Trim(k, `"`)) {
			continue
		}
		// naive replace of "key":"value" → "key":"***"
		for _, quote := range []string{`"`, `'`} {
			_ = quote
		}
		out = maskKey(out, strings.Trim(k, `"`))
	}
	return out
}

func maskKey(s, key string) string {
	// Case-insensitive search for "key" then mask following JSON string value.
	lower := strings.ToLower(s)
	needle := `"` + strings.ToLower(key) + `"`
	var b strings.Builder
	i := 0
	for {
		idx := strings.Index(lower[i:], needle)
		if idx < 0 {
			b.WriteString(s[i:])
			break
		}
		idx += i
		b.WriteString(s[i:idx])
		b.WriteString(s[idx : idx+len(needle)])
		rest := s[idx+len(needle):]
		// skip whitespace and colon
		j := 0
		for j < len(rest) && (rest[j] == ' ' || rest[j] == '\t' || rest[j] == '\n' || rest[j] == '\r') {
			j++
		}
		if j < len(rest) && rest[j] == ':' {
			j++
			for j < len(rest) && (rest[j] == ' ' || rest[j] == '\t') {
				j++
			}
			b.WriteString(rest[:j])
			if j < len(rest) && rest[j] == '"' {
				j++
				for j < len(rest) {
					if rest[j] == '\\' && j+1 < len(rest) {
						j += 2
						continue
					}
					if rest[j] == '"' {
						j++
						break
					}
					j++
				}
				b.WriteString(`"***"`)
				i = idx + len(needle) + j
				continue
			}
		}
		i = idx + len(needle)
	}
	return b.String()
}
