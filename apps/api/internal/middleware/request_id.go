package middleware

import (
	"bytes"
	"encoding/json"
	"errors"
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
	// Hard limit for ordinary JSON bodies (including chunked / ContentLength=-1); exceeded → 413.
	MaxRequestBodyBytes = 32 << 20 // 32 MiB
	// Admin PDF upload has its own ceiling so a valid document is not rejected
	// by the JSON limit. The handler enforces the configured DOCUMENT_MAX_BYTES,
	// which must be at or below DocumentUploadHardMax.
	DocumentUploadHardMax = 64 << 20
	// Multipart boundaries and text fields sit outside the PDF byte count.
	// The request ceiling includes this so a file of exactly the hard max
	// is not rejected as 413.
	DocumentUploadMultipartOverhead = 256 << 10
)

var errPayloadTooLarge = errors.New("request body exceeds size limit")

// RequestID ensures every request has a UUID request_id (header + Gin context).
func RequestID() gin.HandlerFunc {
	return func(c *gin.Context) {
		rid := strings.TrimSpace(c.GetHeader(HeaderRequestID))
		if rid == "" {
			rid = uuid.NewString()
			c.Set("request_id_generated", true)
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

// RequestIDWasGenerated reports whether middleware minted the request id.
func RequestIDWasGenerated(c *gin.Context) bool {
	v, _ := c.Get("request_id_generated")
	b, _ := v.(bool)
	return b
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
	if w.body != nil && w.body.Len() < maxBodyLogBytes {
		remain := maxBodyLogBytes - w.body.Len()
		if len(b) > remain {
			_, _ = w.body.Write(b[:remain])
		} else {
			_, _ = w.body.Write(b)
		}
	}
	return w.ResponseWriter.Write(b)
}

// SensitivePath reports routes whose bodies must never be read or logged.
func SensitivePath(path string) bool {
	p := strings.ToLower(path)
	if strings.Contains(p, "/auth/") {
		return true
	}
	if strings.Contains(p, "/sessions") {
		return true
	}
	if strings.Contains(p, "/admin/documents") {
		return true
	}
	return false
}

func requestBodyLimit(path string) int64 {
	if strings.Contains(strings.ToLower(path), "/admin/documents") {
		return DocumentUploadHardMax + DocumentUploadMultipartOverhead
	}
	return MaxRequestBodyBytes
}

// limitedPreview captures at most maxBodyLogBytes while bytes stream to the handler.
type limitedPreview struct {
	buf *bytes.Buffer
	n   int
}

func (l *limitedPreview) Write(p []byte) (int, error) {
	if l.n >= maxBodyLogBytes {
		return len(p), nil
	}
	remain := maxBodyLogBytes - l.n
	if len(p) > remain {
		_, _ = l.buf.Write(p[:remain])
		l.n = maxBodyLogBytes
		return len(p), nil
	}
	_, _ = l.buf.Write(p)
	l.n += len(p)
	return len(p), nil
}

type teeReadCloser struct {
	io.Reader
	io.Closer
}

// sizeLimitedBody enforces MaxRequestBodyBytes on the actual byte stream
// (Content-Length known or chunked / ContentLength=-1).
type sizeLimitedBody struct {
	r      io.ReadCloser
	remain int64
	guard  *payloadGuard
	rid    string
	c      *gin.Context
}

func (s *sizeLimitedBody) Read(p []byte) (int, error) {
	if s.guard.exceeded {
		return 0, errPayloadTooLarge
	}
	if s.remain <= 0 {
		var one [1]byte
		n, err := s.r.Read(one[:])
		if n > 0 {
			s.guard.markExceeded(s.c, s.rid)
			return 0, errPayloadTooLarge
		}
		if err != nil {
			return 0, err
		}
		return 0, io.EOF
	}
	if int64(len(p)) > s.remain {
		p = p[:s.remain]
	}
	n, err := s.r.Read(p)
	s.remain -= int64(n)
	return n, err
}

func (s *sizeLimitedBody) Close() error {
	return s.r.Close()
}

// payloadGuard forces HTTP 413 when the body limit is exceeded, even if a
// handler would otherwise write 400/500 after a read error.
type payloadGuard struct {
	gin.ResponseWriter
	exceeded bool
	sent413  bool
	logBody  *bytes.Buffer // optional response log capture (non-sensitive)
}

func (p *payloadGuard) markExceeded(c *gin.Context, rid string) {
	p.exceeded = true
	c.Abort()
	if p.sent413 {
		return
	}
	p.sent413 = true
	body := []byte(`{"request_id":"` + rid + `","error":{"code":"PAYLOAD_TOO_LARGE","message":"request body exceeds size limit"}}`)
	p.ResponseWriter.Header().Set("Content-Type", "application/json; charset=utf-8")
	p.ResponseWriter.WriteHeader(http.StatusRequestEntityTooLarge)
	_, _ = p.ResponseWriter.Write(body)
	if p.logBody != nil && p.logBody.Len() < maxBodyLogBytes {
		_, _ = p.logBody.Write(body)
	}
}

func (p *payloadGuard) WriteHeader(code int) {
	if p.exceeded {
		if !p.sent413 {
			// Should already be sent; keep status locked to 413.
			p.ResponseWriter.WriteHeader(http.StatusRequestEntityTooLarge)
		}
		return
	}
	p.ResponseWriter.WriteHeader(code)
}

func (p *payloadGuard) Write(b []byte) (int, error) {
	if p.exceeded {
		// Swallow handler error bodies after we already committed 413.
		return len(b), nil
	}
	if p.logBody != nil && p.logBody.Len() < maxBodyLogBytes {
		remain := maxBodyLogBytes - p.logBody.Len()
		if len(b) > remain {
			_, _ = p.logBody.Write(b[:remain])
		} else {
			_, _ = p.logBody.Write(b)
		}
	}
	return p.ResponseWriter.Write(b)
}

func (p *payloadGuard) Status() int {
	if p.exceeded {
		return http.StatusRequestEntityTooLarge
	}
	return p.ResponseWriter.Status()
}

func write413Early(c *gin.Context, rid string) {
	c.AbortWithStatusJSON(http.StatusRequestEntityTooLarge, gin.H{
		"request_id": rid,
		"error": gin.H{
			"code":    "PAYLOAD_TOO_LARGE",
			"message": "request body exceeds size limit",
		},
	})
}

// ApiLog writes a structured access line. Sensitive routes never log bodies
// but still enforce MaxRequestBodyBytes → HTTP 413 (including chunked bodies).
// Any request that carries a body is limited regardless of method (GET/DELETE/…).
// Non-sensitive: TeeReader log preview ≤4KiB; handler always receives full body up to the limit.
func ApiLog(logger *slog.Logger) gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		rid := GetRequestID(c)
		reqLog := logger.With("request_id", rid)
		c.Set(ContextLogger, reqLog)

		sensitive := SensitivePath(c.Request.URL.Path)
		var previewBuf bytes.Buffer
		var respLogBuf bytes.Buffer
		guard := &payloadGuard{ResponseWriter: c.Writer}
		if !sensitive {
			guard.logBody = &respLogBuf
		}
		c.Writer = guard

		// Enforce on every present body — do not skip by HTTP method.
		if c.Request.Body != nil {
			limit := requestBodyLimit(c.Request.URL.Path)
			if c.Request.ContentLength > limit {
				write413Early(c, rid)
				return
			}
			limited := &sizeLimitedBody{
				r:      c.Request.Body,
				remain: limit,
				guard:  guard,
				rid:    rid,
				c:      c,
			}
			if sensitive {
				// Size-limit only — never tee/log sensitive content.
				c.Request.Body = limited
			} else {
				lp := &limitedPreview{buf: &previewBuf}
				c.Request.Body = &teeReadCloser{
					Reader: io.TeeReader(limited, lp),
					Closer: limited,
				}
			}
		}

		c.Next()

		status := guard.Status()
		latency := time.Since(start).Milliseconds()
		attrs := []any{
			"msg_type", "http_access",
			"request_id", rid,
			"method", c.Request.Method,
			"path", c.Request.URL.Path,
			"status", status,
			"latency_ms", latency,
			"client_ip", c.ClientIP(),
			"bytes_out", c.Writer.Size(),
		}
		if action := c.GetString("log_action"); action != "" {
			attrs = append(attrs, "action", action)
		}

		if sensitive {
			attrs = append(attrs,
				"request_body", "[redacted]",
				"response_body", "[redacted]",
			)
		} else {
			attrs = append(attrs,
				"request_body", truncateBody(RedactSecrets(previewBuf.String())),
				"response_body", truncateBody(RedactSecrets(respLogBuf.String())),
			)
		}

		switch {
		case status >= 500:
			reqLog.Error("http_request", attrs...)
		case status >= 400:
			reqLog.Warn("http_request", attrs...)
		default:
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

var secretKeys = []string{
	"password", "password_hash", "token", "guest_token", "access_token",
	"refresh_token", "secret", "jwt", "authorization", "api_key",
}

// RedactSecrets masks known secret fields in JSON and Authorization-like substrings.
func RedactSecrets(s string) string {
	if s == "" {
		return s
	}
	out := s
	for _, key := range secretKeys {
		out = maskKey(out, key)
	}
	lower := strings.ToLower(out)
	var bearer strings.Builder
	bi := 0
	for {
		idx := strings.Index(lower[bi:], "bearer ")
		if idx < 0 {
			bearer.WriteString(out[bi:])
			break
		}
		idx += bi
		bearer.WriteString(out[bi:idx])
		end := idx + len("bearer ")
		for end < len(out) && !strings.ContainsRune(" \t\n\r\"',}", rune(out[end])) {
			end++
		}
		bearer.WriteString("Bearer ***")
		bi = end
	}
	out = bearer.String()
	var m map[string]any
	if json.Unmarshal([]byte(out), &m) == nil {
		redactMap(m)
		if b, err := json.Marshal(m); err == nil {
			out = string(b)
		}
	}
	return out
}

func redactMap(m map[string]any) {
	for k, v := range m {
		lk := strings.ToLower(k)
		for _, sk := range secretKeys {
			if lk == sk || strings.Contains(lk, sk) {
				m[k] = "***"
				continue
			}
		}
		switch child := v.(type) {
		case map[string]any:
			redactMap(child)
		case []any:
			for _, item := range child {
				if cm, ok := item.(map[string]any); ok {
					redactMap(cm)
				}
			}
		}
	}
}

func maskKey(s, key string) string {
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
