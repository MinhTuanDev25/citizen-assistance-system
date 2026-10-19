package retrieve

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"
	"time"
)

const (
	SchemaVersion           = "retrieve.v1"
	DefaultMaxResponseBytes = 128 << 10
	HardMaxTimeout          = 10 * time.Second
	DefaultTopK             = 8
	MaxTopK                 = 20
	MaxAllowedSources       = 256
	// MinRelatedCosine is the noise floor for L2-normalized cosine, matching
	// Python retrieve.MIN_RELATED_COSINE. It is not supporting proof.
	MinRelatedCosine = 0.10
)

var (
	ErrDisabled        = errors.New("retrieve: client not configured")
	ErrInvalidResponse = errors.New("retrieve: invalid response")
	ErrUnavailable     = errors.New("retrieve: unavailable")
)

// HTTPError is a non-200 retrieve response. Reason is a sanitized code from
// FastAPI detail.reason when present (embedding_unavailable, timeout, …).
type HTTPError struct {
	Status int
	Reason string
}

func (e *HTTPError) Error() string {
	if e == nil {
		return ErrUnavailable.Error()
	}
	if e.Reason != "" {
		return fmt.Sprintf("%s: status %d: %s", ErrUnavailable, e.Status, e.Reason)
	}
	return fmt.Sprintf("%s: status %d", ErrUnavailable, e.Status)
}

func (e *HTTPError) Unwrap() error { return ErrUnavailable }

func UnavailableReason(err error) string {
	var httpErr *HTTPError
	if errors.As(err, &httpErr) {
		return httpErr.Reason
	}
	return ""
}

func PassesRelatedFloor(score *float64) bool {
	if score == nil || math.IsNaN(*score) || math.IsInf(*score, 0) {
		return false
	}
	return *score >= MinRelatedCosine
}

func sanitizeReason(value string) string {
	text := strings.TrimSpace(value)
	if text == "" || len(text) > 64 {
		return ""
	}
	for _, r := range text {
		ok := r == '_' || (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9')
		if !ok {
			return ""
		}
	}
	return strings.ToLower(text)
}

func parseHTTPError(status int, raw []byte) error {
	reason := ""
	var body struct {
		Detail json.RawMessage `json:"detail"`
	}
	if json.Unmarshal(raw, &body) == nil && len(body.Detail) > 0 {
		var obj struct {
			Reason string `json:"reason"`
		}
		if json.Unmarshal(body.Detail, &obj) == nil {
			reason = sanitizeReason(obj.Reason)
		}
	}
	return &HTTPError{Status: status, Reason: reason}
}

type Source struct {
	DocumentID   string `json:"document_id"`
	GenerationID string `json:"generation_id"`
}

type Request struct {
	SchemaVersion      string   `json:"schema_version"`
	RequestID          string   `json:"request_id,omitempty"`
	Query              string   `json:"query"`
	XaID               string   `json:"xa_id"`
	ProcedureID        string   `json:"procedure_id,omitempty"`
	ProcedureVersionID string   `json:"procedure_version_id"`
	AllowedSources     []Source `json:"allowed_sources"`
	TopK               int      `json:"top_k"`
	Statements         []string `json:"statements"`
}

type Hit struct {
	ChunkID            string   `json:"chunk_id"`
	DocumentID         string   `json:"document_id"`
	GenerationID       string   `json:"generation_id"`
	ProcedureVersionID string   `json:"procedure_version_id"`
	Title              string   `json:"title"`
	DocumentNumber     string   `json:"document_number"`
	PageStart          int      `json:"page_start"`
	PageEnd            int      `json:"page_end"`
	Excerpt            string   `json:"excerpt"`
	ExcerptSHA256      string   `json:"excerpt_sha256"`
	ContentSHA256      string   `json:"content_sha256"`
	RetrievalScore     *float64 `json:"retrieval_score"`
	EvidenceStatus     string   `json:"evidence_status"`
	Availability       string   `json:"availability"`
}

type Response struct {
	SchemaVersion string `json:"schema_version"`
	Status        string `json:"status"`
	Hits          []Hit  `json:"hits"`
	Reason        string `json:"reason"`
}

type Retriever interface {
	Retrieve(ctx context.Context, req Request) (*Response, error)
}

type Client struct {
	BaseURL          string
	HTTPClient       *http.Client
	Timeout          time.Duration
	MaxResponseBytes int64
	ServiceToken     string
}

func NewClient(baseURL string, timeout time.Duration) *Client {
	if timeout <= 0 || timeout > HardMaxTimeout {
		timeout = HardMaxTimeout
	}
	return &Client{
		BaseURL:          strings.TrimRight(baseURL, "/"),
		HTTPClient:       &http.Client{Timeout: timeout},
		Timeout:          timeout,
		MaxResponseBytes: DefaultMaxResponseBytes,
	}
}

func (c *Client) Retrieve(ctx context.Context, req Request) (*Response, error) {
	if c == nil || c.BaseURL == "" {
		return nil, ErrDisabled
	}
	req.SchemaVersion = SchemaVersion
	if req.TopK < 1 || req.TopK > MaxTopK {
		req.TopK = DefaultTopK
	}
	if req.AllowedSources == nil {
		req.AllowedSources = []Source{}
	}
	if req.Statements == nil {
		req.Statements = []string{}
	}
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("retrieve: marshal: %w", err)
	}
	timeout := c.Timeout
	if timeout <= 0 || timeout > HardMaxTimeout {
		timeout = HardMaxTimeout
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+"/v1/retrieve", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("retrieve: build: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if req.RequestID != "" {
		httpReq.Header.Set("X-Request-ID", req.RequestID)
	}
	if c.ServiceToken != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.ServiceToken)
	}
	client := c.HTTPClient
	if client == nil {
		client = &http.Client{Timeout: timeout}
	}
	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrUnavailable, err)
	}
	defer resp.Body.Close()
	maxBytes := c.MaxResponseBytes
	if maxBytes <= 0 {
		maxBytes = DefaultMaxResponseBytes
	}
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, maxBytes))
		return nil, parseHTTPError(resp.StatusCode, raw)
	}
	raw, err := io.ReadAll(io.LimitReader(resp.Body, maxBytes+1))
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrUnavailable, err)
	}
	if int64(len(raw)) > maxBytes {
		return nil, fmt.Errorf("%w: response too large", ErrUnavailable)
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	var out Response
	if err := dec.Decode(&out); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalidResponse, err)
	}
	var trailing json.RawMessage
	if err := dec.Decode(&trailing); err != io.EOF {
		return nil, fmt.Errorf("%w: trailing data", ErrInvalidResponse)
	}
	if out.SchemaVersion != SchemaVersion {
		return nil, fmt.Errorf("%w: schema", ErrInvalidResponse)
	}
	switch out.Status {
	case "ok", "no_relevant_evidence", "all_candidates_rejected", "retrieval_unavailable", "retrieval_incompatible":
	default:
		return nil, fmt.Errorf("%w: status", ErrInvalidResponse)
	}
	if out.Hits == nil {
		out.Hits = []Hit{}
	}
	return &out, nil
}
