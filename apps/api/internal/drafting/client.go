package drafting

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/decision"
)

const (
	DefaultDraftMaxResponseBytes = 512 << 10
	HardMaxDraftTimeout          = 120 * time.Second
	DefaultDraftTimeout          = 45 * time.Second
)

var (
	ErrDisabled     = errors.New("drafting: client not configured")
	ErrRateLimited  = errors.New("drafting: rate limited")
	ErrInvalidDraft = errors.New("drafting: invalid provider response")
)

type Client struct {
	BaseURL          string
	HTTPClient       *http.Client
	Timeout          time.Duration
	MaxResponseBytes int64
	ServiceToken     string
}

type providerRequest struct {
	SchemaVersion   string    `json:"schema_version"`
	RequestID       string    `json:"request_id"`
	ProcedureCode   string    `json:"procedure_code"`
	ProcedureName   string    `json:"procedure_name"`
	XaID            string    `json:"xa_id"`
	DocumentID      string    `json:"document_id"`
	GenerationID    string    `json:"generation_id"`
	SourcePageStart int       `json:"source_page_start"`
	SourcePageEnd   int       `json:"source_page_end"`
	Excerpts        []Excerpt `json:"excerpts"`
}

type ProviderResponse struct {
	SchemaVersion       string          `json:"schema_version"`
	RequestID           string          `json:"request_id"`
	ProcedureCode       string          `json:"procedure_code"`
	Definition          json.RawMessage `json:"definition"`
	Evidence            []EvidenceItem  `json:"evidence"`
	Warnings            []string        `json:"warnings"`
	MissingRequirements []string        `json:"missing_requirements"`
	Validation          struct {
		SchemaValid       bool                  `json:"schema_valid"`
		GroundingVerified bool                  `json:"grounding_verified"`
		SemanticApproved  bool                  `json:"semantic_approved"`
		Issues            []decision.DraftIssue `json:"issues"`
		Readiness         string                `json:"readiness"`
	} `json:"validation"`
	Provider              string         `json:"provider"`
	Model                 string         `json:"model"`
	PromptVersion         string         `json:"prompt_version"`
	PromptHash            string         `json:"prompt_hash"`
	Usage                 map[string]any `json:"usage"`
	LatencyMS             int            `json:"latency_ms"`
	HTTPStatus            int            `json:"http_status"`
	AbstainReason         string         `json:"abstain_reason"`
	OriginalProcedureCode string         `json:"original_procedure_code"`
	IdentityMismatch      bool           `json:"identity_mismatch"`
}

func NewClient(baseURL string, timeout time.Duration) *Client {
	if timeout <= 0 || timeout > HardMaxDraftTimeout {
		timeout = DefaultDraftTimeout
	}
	return &Client{
		BaseURL:          strings.TrimRight(baseURL, "/"),
		HTTPClient:       &http.Client{Timeout: timeout},
		Timeout:          timeout,
		MaxResponseBytes: DefaultDraftMaxResponseBytes,
	}
}

func (c *Client) Draft(ctx context.Context, requestID uuid.UUID, snap SourceSnapshot) (*ProviderResponse, error) {
	if c == nil || c.BaseURL == "" {
		return nil, ErrDisabled
	}
	payload := providerRequest{
		SchemaVersion:   "draft_definition.v1",
		RequestID:       requestID.String(),
		ProcedureCode:   snap.ProcedureCode,
		ProcedureName:   snap.ProcedureName,
		XaID:            snap.XaID,
		DocumentID:      snap.DocumentID.String(),
		GenerationID:    snap.GenerationID.String(),
		SourcePageStart: snap.SourcePageStart,
		SourcePageEnd:   snap.SourcePageEnd,
		Excerpts:        snap.Excerpts,
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("drafting: marshal: %w", err)
	}
	timeout := c.Timeout
	if timeout <= 0 || timeout > HardMaxDraftTimeout {
		timeout = HardMaxDraftTimeout
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+"/v1/draft_definition", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("X-Request-ID", requestID.String())
	if c.ServiceToken != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.ServiceToken)
	}
	client := c.HTTPClient
	if client == nil {
		client = &http.Client{Timeout: timeout}
	}
	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("drafting: request failed: %w", err)
	}
	defer resp.Body.Close()
	maxBytes := c.MaxResponseBytes
	if maxBytes <= 0 {
		maxBytes = DefaultDraftMaxResponseBytes
	}
	if resp.StatusCode == http.StatusTooManyRequests {
		_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, maxBytes))
		return nil, ErrRateLimited
	}
	raw, err := io.ReadAll(io.LimitReader(resp.Body, maxBytes+1))
	if err != nil {
		return nil, fmt.Errorf("drafting: read response: %w", err)
	}
	if int64(len(raw)) > maxBytes {
		return nil, fmt.Errorf("drafting: response exceeds %d bytes", maxBytes)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("drafting: unexpected status %d", resp.StatusCode)
	}
	var out ProviderResponse
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalidDraft, err)
	}
	if out.SchemaVersion != "draft_definition.v1" {
		return nil, fmt.Errorf("%w: unexpected schema_version", ErrInvalidDraft)
	}
	return &out, nil
}
