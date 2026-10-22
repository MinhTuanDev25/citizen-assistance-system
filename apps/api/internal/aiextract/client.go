package aiextract

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
)

// DefaultMaxResponseBytes bounds how much of the AI service's response body
// the Go side will ever read, regardless of what Content-Length claims.
const DefaultMaxResponseBytes = 64 << 10 // 64 KiB

// HardMaxTimeout is an upper bound on the per-call timeout regardless of
// configuration, so a misconfigured AI_EXTRACT_TIMEOUT_MS can never make the
// citizen-facing turn hang indefinitely.
const HardMaxTimeout = 8 * time.Second

// ErrDisabled is returned when the client has no base URL configured.
var ErrDisabled = errors.New("aiextract: client not configured")

// ErrInvalidResponse is a contract failure (unknown field, trailing data,
// wrong schema). Callers fall back to keyword matching and must not persist
// any field from the body.
var ErrInvalidResponse = errors.New("aiextract: invalid response")

// ErrRateLimited is HTTP 429. The client does not retry; the caller falls
// back to keyword matching.
var ErrRateLimited = errors.New("aiextract: rate limited")

// Client is a single-attempt, bounded-timeout HTTP client for POST /v1/extract.
// It never retries — the caller (chat.Service) is responsible for the
// keyword fallback on any error.
type Client struct {
	BaseURL          string
	HTTPClient       *http.Client
	Timeout          time.Duration
	MaxResponseBytes int64
	// ServiceToken is sent as "Authorization: Bearer …". It is never logged.
	ServiceToken string
}

// NewClient builds a Client with a timeout clamped to (0, HardMaxTimeout].
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

// Extract calls POST {BaseURL}/v1/extract. It never returns a partially
// decoded Response on error — callers must treat any error as "no
// extraction happened" and fall back to keyword matching.
func (c *Client) Extract(ctx context.Context, req Request) (*Response, error) {
	if c == nil || c.BaseURL == "" {
		return nil, ErrDisabled
	}
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("aiextract: marshal request: %w", err)
	}

	timeout := c.Timeout
	if timeout <= 0 || timeout > HardMaxTimeout {
		timeout = HardMaxTimeout
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+"/v1/extract", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("aiextract: build request: %w", err)
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
		return nil, fmt.Errorf("aiextract: request failed: %w", err)
	}
	defer resp.Body.Close()

	maxBytes := c.MaxResponseBytes
	if maxBytes <= 0 {
		maxBytes = DefaultMaxResponseBytes
	}

	if resp.StatusCode == http.StatusTooManyRequests {
		_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, maxBytes))
		return nil, ErrRateLimited
	}
	if resp.StatusCode != http.StatusOK {
		_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, maxBytes))
		return nil, fmt.Errorf("aiextract: unexpected status %d", resp.StatusCode)
	}

	limited := io.LimitReader(resp.Body, maxBytes+1)
	raw, err := io.ReadAll(limited)
	if err != nil {
		return nil, fmt.Errorf("aiextract: read response: %w", err)
	}
	if int64(len(raw)) > maxBytes {
		return nil, fmt.Errorf("aiextract: response exceeds %d bytes", maxBytes)
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
		return nil, fmt.Errorf("%w: unexpected schema_version", ErrInvalidResponse)
	}
	return &out, nil
}
