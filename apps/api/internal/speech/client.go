// Package speech is the Go HTTP client for the internal C0 Speech Inference API.
package speech

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"strings"
	"time"
)

const (
	InferPath               = "/v1/speech/infer"
	RequiredModelID         = "c0_cascaded"
	RequiredTokenizerParity = "VERIFIED"
	DefaultMaxResponseBytes = 64 << 10
	HardMaxTimeout          = 180 * time.Second
	// UpstreamBodyHardMax is FastAPI SPEECH_MAX_UPLOAD_BYTES: the whole
	// re-encoded multipart body, not the WAV part alone.
	UpstreamBodyHardMax = 10 << 20
	// MultipartOverheadReserve is a conservative margin above measured
	// MIME headers (~a few hundred bytes for field "audio"). It is not
	// claimed to be exact; EncodedMultipartLen is the authority.
	MultipartOverheadReserve int64 = 64 << 10
)

var errNoRedirect = errors.New("speech: refusing HTTP redirect")

var (
	ErrDisabled        = errors.New("speech: client not configured")
	ErrCanceled        = errors.New("speech: request canceled")
	ErrTimeout         = errors.New("speech: inference timeout")
	ErrUnavailable     = errors.New("speech: service unavailable")
	ErrNotReady        = errors.New("speech: service not ready")
	ErrOverloaded      = errors.New("speech: inference overloaded")
	ErrUpstreamAuth    = errors.New("speech: upstream authentication failed")
	ErrInvalidAudio    = errors.New("speech: invalid audio")
	ErrTooLarge        = errors.New("speech: payload too large")
	ErrInvalidResponse = errors.New("speech: invalid upstream response")
	ErrInferenceFailed = errors.New("speech: inference failed")
	ErrNoSpeech        = errors.New("speech: no speech recognised")
	ErrShuttingDown    = errors.New("speech: service shutting down")
)

// Translator is the inference dependency used by the HTTP handler.
type Translator interface {
	Infer(ctx context.Context, req InferRequest) (*Result, error)
}

type InferRequest struct {
	Audio     []byte
	Filename  string
	RequestID string
}

// Result is the decoded upstream payload. Public handlers must project a subset.
type Result struct {
	RequestID            string
	ModelID              string
	AsrTextBahnar        string
	TranslatedTextVi     string
	DurationMS           int
	LatencyMS            int
	TokenizerParity      string
	ResearchProvenance   string
	CheckpointIdentity   string
	GenerationMatchesRQ1 *bool
	TokenizerSource      string
	TokenizerModelID     string
	TokenizerRevision    string
	ASRCheckpointStep    *int
	MTCheckpointStep     *int
	ArtifactsPresent     *bool
	UpstreamStatus       int
}

type Client struct {
	BaseURL          string
	HTTPClient       *http.Client
	Timeout          time.Duration
	MaxResponseBytes int64
	MaxUpstreamBody  int64
	ServiceToken     string
}

func NewClient(baseURL string, timeout time.Duration) *Client {
	if timeout <= 0 || timeout > HardMaxTimeout {
		timeout = HardMaxTimeout
	}
	transport := http.DefaultTransport.(*http.Transport).Clone()
	return &Client{
		BaseURL:          strings.TrimRight(strings.TrimSpace(baseURL), "/"),
		HTTPClient:       newHTTPClient(timeout, transport),
		Timeout:          timeout,
		MaxResponseBytes: DefaultMaxResponseBytes,
		MaxUpstreamBody:  UpstreamBodyHardMax,
	}
}

func newHTTPClient(timeout time.Duration, transport http.RoundTripper) *http.Client {
	return &http.Client{
		Timeout:   timeout,
		Transport: transport,
		CheckRedirect: func(*http.Request, []*http.Request) error {
			return errNoRedirect
		},
	}
}

// MaxAudioBytes is the largest WAV the handler may accept so the re-encoded
// multipart body stays at or under the FastAPI whole-body cap.
// Limits at or below MultipartOverheadReserve return 0 (no fail-open to 10 MiB).
func MaxAudioBytes(upstreamBodyLimit int64) int64 {
	if upstreamBodyLimit > UpstreamBodyHardMax {
		upstreamBodyLimit = UpstreamBodyHardMax
	}
	if upstreamBodyLimit <= MultipartOverheadReserve {
		return 0
	}
	return upstreamBodyLimit - MultipartOverheadReserve
}

// EffectiveUpstreamBody is the re-encoded multipart ceiling: the configured
// body cap, never above UpstreamBodyHardMax, never substituted upward from 0.
func EffectiveUpstreamBody(limit int64) int64 {
	if limit <= 0 {
		return 0
	}
	if limit > UpstreamBodyHardMax {
		return UpstreamBodyHardMax
	}
	return limit
}

// EncodeAudioMultipart builds the same multipart body Infer sends upstream.
func EncodeAudioMultipart(audio []byte, filename string) ([]byte, string, error) {
	filename = strings.TrimSpace(filename)
	if filename == "" {
		filename = "audio.wav"
	}
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	part, err := writer.CreateFormFile("audio", filename)
	if err != nil {
		return nil, "", fmt.Errorf("speech: multipart: %w", err)
	}
	if _, err := part.Write(audio); err != nil {
		return nil, "", fmt.Errorf("speech: write audio: %w", err)
	}
	if err := writer.Close(); err != nil {
		return nil, "", fmt.Errorf("speech: close multipart: %w", err)
	}
	return body.Bytes(), writer.FormDataContentType(), nil
}

func EncodedMultipartLen(audio []byte, filename string) (int, error) {
	raw, _, err := EncodeAudioMultipart(audio, filename)
	if err != nil {
		return 0, err
	}
	return len(raw), nil
}

func (c *Client) Infer(ctx context.Context, req InferRequest) (*Result, error) {
	if c == nil || c.BaseURL == "" {
		return nil, ErrDisabled
	}
	timeout := c.Timeout
	if timeout <= 0 || timeout > HardMaxTimeout {
		timeout = HardMaxTimeout
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	raw, contentType, err := EncodeAudioMultipart(req.Audio, req.Filename)
	if err != nil {
		return nil, err
	}
	limit := EffectiveUpstreamBody(c.MaxUpstreamBody)
	if limit <= 0 || int64(len(raw)) > limit {
		return nil, ErrTooLarge
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+InferPath, bytes.NewReader(raw))
	if err != nil {
		return nil, fmt.Errorf("speech: build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", contentType)
	if req.RequestID != "" {
		httpReq.Header.Set("X-Request-ID", req.RequestID)
	}
	if c.ServiceToken != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.ServiceToken)
	}

	client := c.HTTPClient
	if client == nil {
		client = newHTTPClient(timeout, nil)
	}
	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, mapTransportError(ctx, err)
	}
	defer resp.Body.Close()

	maxBytes := c.MaxResponseBytes
	if maxBytes <= 0 {
		maxBytes = DefaultMaxResponseBytes
	}
	respRaw, err := io.ReadAll(io.LimitReader(resp.Body, maxBytes+1))
	if err != nil {
		return nil, fmt.Errorf("speech: read response: %w", err)
	}
	if int64(len(respRaw)) > maxBytes {
		return nil, ErrInvalidResponse
	}
	if resp.StatusCode != http.StatusOK {
		return nil, mapStatus(resp.StatusCode, respRaw)
	}

	var wire upstreamBody
	dec := json.NewDecoder(bytes.NewReader(respRaw))
	if err := dec.Decode(&wire); err != nil {
		return nil, fmt.Errorf("%w", ErrInvalidResponse)
	}
	var trailing json.RawMessage
	if err := dec.Decode(&trailing); err != io.EOF {
		return nil, fmt.Errorf("%w", ErrInvalidResponse)
	}
	out, verr := wire.result()
	if verr != nil {
		return nil, verr
	}
	out.UpstreamStatus = resp.StatusCode
	return out, nil
}

type upstreamBody struct {
	RequestID            string  `json:"request_id"`
	ModelID              *string `json:"model_id"`
	AsrTextBahnar        *string `json:"asr_text_bahnar"`
	TranslatedTextVi     *string `json:"translated_text_vi"`
	DurationMS           *int    `json:"duration_ms"`
	LatencyMS            *int    `json:"latency_ms"`
	TokenizerParity      *string `json:"tokenizer_parity"`
	ResearchProvenance   string  `json:"research_provenance"`
	CheckpointIdentity   string  `json:"checkpoint_identity"`
	GenerationMatchesRQ1 *bool   `json:"generation_matches_rq1"`
	TokenizerSource      string  `json:"tokenizer_source"`
	TokenizerModelID     string  `json:"tokenizer_model_id"`
	TokenizerRevision    string  `json:"tokenizer_revision"`
	ASRCheckpointStep    *int    `json:"asr_checkpoint_step"`
	MTCheckpointStep     *int    `json:"mt_checkpoint_step"`
	ArtifactsPresent     *bool   `json:"artifacts_present"`
	Error                *struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
}

func (u upstreamBody) result() (*Result, error) {
	if u.ModelID == nil || strings.TrimSpace(*u.ModelID) != RequiredModelID {
		return nil, ErrInvalidResponse
	}
	if u.AsrTextBahnar == nil || strings.TrimSpace(*u.AsrTextBahnar) == "" {
		return nil, ErrInvalidResponse
	}
	if u.TranslatedTextVi == nil || strings.TrimSpace(*u.TranslatedTextVi) == "" {
		return nil, ErrInvalidResponse
	}
	if u.DurationMS == nil || *u.DurationMS < 0 {
		return nil, ErrInvalidResponse
	}
	if u.LatencyMS == nil || *u.LatencyMS < 0 {
		return nil, ErrInvalidResponse
	}
	if u.TokenizerParity == nil || strings.TrimSpace(*u.TokenizerParity) != RequiredTokenizerParity {
		return nil, ErrInvalidResponse
	}
	if u.GenerationMatchesRQ1 == nil || !*u.GenerationMatchesRQ1 {
		return nil, ErrInvalidResponse
	}
	return &Result{
		RequestID:            u.RequestID,
		ModelID:              strings.TrimSpace(*u.ModelID),
		AsrTextBahnar:        *u.AsrTextBahnar,
		TranslatedTextVi:     *u.TranslatedTextVi,
		DurationMS:           *u.DurationMS,
		LatencyMS:            *u.LatencyMS,
		TokenizerParity:      strings.TrimSpace(*u.TokenizerParity),
		ResearchProvenance:   u.ResearchProvenance,
		CheckpointIdentity:   u.CheckpointIdentity,
		GenerationMatchesRQ1: u.GenerationMatchesRQ1,
		TokenizerSource:      u.TokenizerSource,
		TokenizerModelID:     u.TokenizerModelID,
		TokenizerRevision:    u.TokenizerRevision,
		ASRCheckpointStep:    u.ASRCheckpointStep,
		MTCheckpointStep:     u.MTCheckpointStep,
		ArtifactsPresent:     u.ArtifactsPresent,
	}, nil
}

type mappedError struct {
	kind     error
	upstream int
	code     string
}

func (e mappedError) Error() string { return e.kind.Error() }
func (e mappedError) Unwrap() error { return e.kind }

func UpstreamStatus(err error) int {
	var mapped mappedError
	if errors.As(err, &mapped) {
		return mapped.upstream
	}
	return 0
}

func UpstreamCode(err error) string {
	var mapped mappedError
	if errors.As(err, &mapped) {
		return mapped.code
	}
	return ""
}

func mapStatus(status int, raw []byte) error {
	code := peekErrorCode(raw)
	// Match empty_asr by upstream code, not HTTP status. Live Speech API
	// images may still return 502 for this case; silence is not SPEECH_UNAVAILABLE.
	if code == "empty_asr" {
		return mappedError{kind: ErrNoSpeech, upstream: status, code: code}
	}
	kind := ErrUnavailable
	switch status {
	case http.StatusBadRequest:
		kind = ErrInvalidAudio
	case http.StatusUnauthorized:
		kind = ErrUpstreamAuth
	case http.StatusRequestEntityTooLarge:
		kind = ErrTooLarge
	case http.StatusUnprocessableEntity:
		kind = ErrInvalidAudio
	case http.StatusGatewayTimeout:
		kind = ErrTimeout
	case http.StatusServiceUnavailable:
		switch code {
		case "overloaded":
			kind = ErrOverloaded
		case "not_ready":
			kind = ErrNotReady
		case "shutting_down":
			kind = ErrShuttingDown
		default:
			kind = ErrUnavailable
		}
	case http.StatusBadGateway, http.StatusInternalServerError:
		kind = ErrInferenceFailed
	}
	return mappedError{kind: kind, upstream: status, code: code}
}

func peekErrorCode(raw []byte) string {
	var body struct {
		Error struct {
			Code string `json:"code"`
		} `json:"error"`
	}
	if json.Unmarshal(raw, &body) != nil {
		return ""
	}
	return strings.TrimSpace(body.Error.Code)
}

func mapTransportError(ctx context.Context, err error) error {
	if errors.Is(err, errNoRedirect) {
		return ErrUnavailable
	}
	if errors.Is(err, context.Canceled) || (ctx.Err() != nil && errors.Is(ctx.Err(), context.Canceled)) {
		return ErrCanceled
	}
	if errors.Is(err, context.DeadlineExceeded) || (ctx.Err() != nil && errors.Is(ctx.Err(), context.DeadlineExceeded)) {
		return ErrTimeout
	}
	msg := strings.ToLower(err.Error())
	if strings.Contains(msg, "timeout") || strings.Contains(msg, "deadline exceeded") {
		return ErrTimeout
	}
	return fmt.Errorf("%w", ErrUnavailable)
}
