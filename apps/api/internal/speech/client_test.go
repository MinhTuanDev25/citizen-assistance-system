package speech

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/config"
)

func TestClientSendsMultipartBearerAndRequestID(t *testing.T) {
	t.Parallel()
	var calls atomic.Int32
	var gotAuth, gotRID, gotPath, gotField string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		gotAuth = r.Header.Get("Authorization")
		gotRID = r.Header.Get("X-Request-ID")
		gotPath = r.URL.Path
		media, params, err := mime.ParseMediaType(r.Header.Get("Content-Type"))
		if err != nil || media != "multipart/form-data" {
			t.Errorf("content-type %s %v", r.Header.Get("Content-Type"), err)
		}
		reader := multipart.NewReader(r.Body, params["boundary"])
		part, err := reader.NextPart()
		if err != nil {
			t.Fatalf("part: %v", err)
		}
		gotField = part.FormName()
		_, _ = io.Copy(io.Discard, part)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(okBody())
	}))
	defer srv.Close()

	c := NewClient(srv.URL, time.Second)
	c.ServiceToken = "speech-service-token-1"
	out, err := c.Infer(context.Background(), InferRequest{
		Audio: []byte("RIFF....WAVE"), Filename: "utt.wav", RequestID: "11111111-1111-1111-1111-111111111111",
	})
	if err != nil {
		t.Fatal(err)
	}
	if calls.Load() != 1 {
		t.Fatalf("retries: %d", calls.Load())
	}
	if gotPath != InferPath || gotField != "audio" {
		t.Fatalf("path=%s field=%s", gotPath, gotField)
	}
	if gotAuth != "Bearer speech-service-token-1" {
		t.Fatalf("auth %q", gotAuth)
	}
	if gotRID != "11111111-1111-1111-1111-111111111111" {
		t.Fatalf("rid %q", gotRID)
	}
	if out.AsrTextBahnar != "xin" || out.TranslatedTextVi != "tôi" {
		t.Fatalf("body %+v", out)
	}
	if strings.Contains(gotAuth, "logged") {
		t.Fatal("token confusion")
	}
}

func TestClientRejectsInvalidServingContract(t *testing.T) {
	t.Parallel()
	cases := map[string]string{
		"malformed": `{`,
		"missing_model": `{
			"asr_text_bahnar":"xin","translated_text_vi":"tôi","duration_ms":1,"latency_ms":1,
			"tokenizer_parity":"VERIFIED","generation_matches_rq1":true
		}`,
		"wrong_model": `{
			"model_id":"other_model","asr_text_bahnar":"abc","translated_text_vi":"xyz",
			"duration_ms":-10,"latency_ms":-3,"tokenizer_parity":"NOT_VERIFIED","generation_matches_rq1":false
		}`,
		"missing_transcript": `{
			"model_id":"c0_cascaded","translated_text_vi":"tôi","duration_ms":1,"latency_ms":1,
			"tokenizer_parity":"VERIFIED","generation_matches_rq1":true
		}`,
		"empty_translation": `{
			"model_id":"c0_cascaded","asr_text_bahnar":"xin","translated_text_vi":"",
			"duration_ms":1,"latency_ms":1,"tokenizer_parity":"VERIFIED","generation_matches_rq1":true
		}`,
		"negative_duration": `{
			"model_id":"c0_cascaded","asr_text_bahnar":"xin","translated_text_vi":"tôi",
			"duration_ms":-1,"latency_ms":1,"tokenizer_parity":"VERIFIED","generation_matches_rq1":true
		}`,
		"negative_latency": `{
			"model_id":"c0_cascaded","asr_text_bahnar":"xin","translated_text_vi":"tôi",
			"duration_ms":1,"latency_ms":-3,"tokenizer_parity":"VERIFIED","generation_matches_rq1":true
		}`,
		"missing_parity": `{
			"model_id":"c0_cascaded","asr_text_bahnar":"xin","translated_text_vi":"tôi",
			"duration_ms":1,"latency_ms":1,"generation_matches_rq1":true
		}`,
		"parity_not_verified": `{
			"model_id":"c0_cascaded","asr_text_bahnar":"xin","translated_text_vi":"tôi",
			"duration_ms":1,"latency_ms":1,"tokenizer_parity":"NOT_VERIFIED","generation_matches_rq1":true
		}`,
		"missing_generation": `{
			"model_id":"c0_cascaded","asr_text_bahnar":"xin","translated_text_vi":"tôi",
			"duration_ms":1,"latency_ms":1,"tokenizer_parity":"VERIFIED"
		}`,
		"generation_false": `{
			"model_id":"c0_cascaded","asr_text_bahnar":"xin","translated_text_vi":"tôi",
			"duration_ms":1,"latency_ms":1,"tokenizer_parity":"VERIFIED","generation_matches_rq1":false
		}`,
		"trailing_json": `{"model_id":"c0_cascaded","asr_text_bahnar":"xin","translated_text_vi":"tôi","duration_ms":1,"latency_ms":1,"tokenizer_parity":"VERIFIED","generation_matches_rq1":true}{"x":1}`,
	}
	for name, body := range cases {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, body)
		}))
		c := NewClient(srv.URL, time.Second)
		_, err := c.Infer(context.Background(), InferRequest{Audio: []byte("x")})
		srv.Close()
		if !errors.Is(err, ErrInvalidResponse) {
			t.Fatalf("%s: err=%v", name, err)
		}
	}
}

func TestClientAcceptsNotVerifiedProvenance(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(okBody())
	}))
	defer srv.Close()
	c := NewClient(srv.URL, time.Second)
	out, err := c.Infer(context.Background(), InferRequest{Audio: []byte("x")})
	if err != nil {
		t.Fatal(err)
	}
	if out.ResearchProvenance != "NOT_VERIFIED" {
		t.Fatalf("rewrote provenance %q", out.ResearchProvenance)
	}
}

func TestClientEnforcesResponseSize(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(bytes.Repeat([]byte("a"), 128))
	}))
	defer srv.Close()
	c := NewClient(srv.URL, time.Second)
	c.MaxResponseBytes = 16
	_, err := c.Infer(context.Background(), InferRequest{Audio: []byte("x")})
	if !errors.Is(err, ErrInvalidResponse) {
		t.Fatalf("err=%v", err)
	}
}

func TestClientMapsStatuses(t *testing.T) {
	t.Parallel()
	table := []struct {
		status int
		body   string
		want   error
	}{
		{400, `{"error":{"code":"audio_invalid"}}`, ErrInvalidAudio},
		{401, `{"error":{"code":"unauthorized"}}`, ErrUpstreamAuth},
		{413, `{"error":{"code":"payload_too_large"}}`, ErrTooLarge},
		{503, `{"error":{"code":"overloaded"}}`, ErrOverloaded},
		{503, `{"error":{"code":"not_ready"}}`, ErrNotReady},
		{504, `{"error":{"code":"inference_timeout"}}`, ErrTimeout},
		{502, `{"error":{"code":"empty_asr"}}`, ErrInferenceFailed},
		{500, `{"error":{"code":"inference_failed"}}`, ErrInferenceFailed},
	}
	for _, tc := range table {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(tc.status)
			_, _ = io.WriteString(w, tc.body)
		}))
		c := NewClient(srv.URL, time.Second)
		_, err := c.Infer(context.Background(), InferRequest{Audio: []byte("x")})
		srv.Close()
		if !errors.Is(err, tc.want) {
			t.Fatalf("status %d: got %v want %v", tc.status, err, tc.want)
		}
		if UpstreamStatus(err) != tc.status {
			t.Fatalf("upstream %d", UpstreamStatus(err))
		}
	}
}

func TestClientDoesNotRetryOn503(t *testing.T) {
	t.Parallel()
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = io.WriteString(w, `{"error":{"code":"overloaded"}}`)
	}))
	defer srv.Close()
	c := NewClient(srv.URL, time.Second)
	_, err := c.Infer(context.Background(), InferRequest{Audio: []byte("x")})
	if !errors.Is(err, ErrOverloaded) {
		t.Fatalf("err=%v", err)
	}
	if calls.Load() != 1 {
		t.Fatalf("calls=%d", calls.Load())
	}
}

func TestClientHonorsCancel(t *testing.T) {
	t.Parallel()
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()
	c := NewClient(srv.URL, 2*time.Second)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := c.Infer(ctx, InferRequest{Audio: []byte("x")})
	if !errors.Is(err, ErrCanceled) && !errors.Is(err, ErrTimeout) && !errors.Is(err, ErrUnavailable) {
		t.Fatalf("cancel err=%v", err)
	}
	if calls.Load() != 0 {
		t.Fatalf("canceled request still hit upstream: %d", calls.Load())
	}
}

func TestClientHonorsTimeout(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(200 * time.Millisecond)
	}))
	defer srv.Close()
	c := NewClient(srv.URL, 30*time.Millisecond)
	c.HTTPClient = &http.Client{Timeout: 30 * time.Millisecond}
	_, err := c.Infer(context.Background(), InferRequest{Audio: []byte("x")})
	if !errors.Is(err, ErrTimeout) && !errors.Is(err, ErrUnavailable) {
		t.Fatalf("timeout err=%v", err)
	}
}

func TestClientDoesNotLogToken(t *testing.T) {
	t.Parallel()
	token := "super-secret-speech-token"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = io.WriteString(w, `{"error":{"code":"unauthorized","message":"nope"}}`)
	}))
	defer srv.Close()
	c := NewClient(srv.URL, time.Second)
	c.ServiceToken = token
	_, err := c.Infer(context.Background(), InferRequest{Audio: []byte("x")})
	if err == nil || strings.Contains(err.Error(), token) {
		t.Fatalf("token leaked: %v", err)
	}
}

func TestClientRefusesRedirects(t *testing.T) {
	t.Parallel()
	var secondAuth string
	final := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		secondAuth = r.Header.Get("Authorization")
		w.WriteHeader(http.StatusOK)
	}))
	defer final.Close()
	redir := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, final.URL, http.StatusFound)
	}))
	defer redir.Close()
	c := NewClient(redir.URL, time.Second)
	c.ServiceToken = "speech-service-token-1"
	_, err := c.Infer(context.Background(), InferRequest{Audio: []byte("x")})
	if !errors.Is(err, ErrUnavailable) {
		t.Fatalf("err=%v", err)
	}
	if secondAuth != "" {
		t.Fatalf("Authorization forwarded to redirect host: %q", secondAuth)
	}
}

func TestEncodedMultipartStaysWithinUpstreamCap(t *testing.T) {
	t.Parallel()
	small := []byte("RIFF....WAVE")
	n, err := EncodedMultipartLen(small, "utt.wav")
	if err != nil {
		t.Fatal(err)
	}
	if n <= len(small) {
		t.Fatalf("expected multipart overhead, got %d for %d audio", n, len(small))
	}
	maxAudio := MaxAudioBytes(UpstreamBodyHardMax)
	audio := make([]byte, maxAudio)
	copy(audio, []byte("RIFF....WAVE"))
	encoded, err := EncodedMultipartLen(audio, "utt.wav")
	if err != nil {
		t.Fatal(err)
	}
	if int64(encoded) > UpstreamBodyHardMax {
		t.Fatalf("encoded %d exceeds upstream %d", encoded, UpstreamBodyHardMax)
	}
	tooBig := make([]byte, UpstreamBodyHardMax)
	copy(tooBig, []byte("RIFF....WAVE"))
	c := NewClient("http://127.0.0.1:9", time.Second)
	_, err = c.Infer(context.Background(), InferRequest{Audio: tooBig})
	if !errors.Is(err, ErrTooLarge) {
		t.Fatalf("exact body-cap WAV must be rejected locally, err=%v", err)
	}
}

func TestUploadLimitConstantsAlign(t *testing.T) {
	t.Parallel()
	if config.SpeechUploadHardMaxBytes != UpstreamBodyHardMax {
		t.Fatalf("hard max config %d client %d", config.SpeechUploadHardMaxBytes, UpstreamBodyHardMax)
	}
	if config.SpeechUploadMinBytes != MultipartOverheadReserve+1 {
		t.Fatalf("min %d reserve %d", config.SpeechUploadMinBytes, MultipartOverheadReserve)
	}
}

func TestMaxAudioBytesDoesNotFailOpen(t *testing.T) {
	t.Parallel()
	if MultipartOverheadReserve != 65536 {
		t.Fatalf("reserve %d", MultipartOverheadReserve)
	}
	for _, n := range []int64{0, 1, 65535, 65536} {
		if got := MaxAudioBytes(n); got != 0 {
			t.Fatalf("MaxAudioBytes(%d)=%d, want 0", n, got)
		}
	}
	if got := MaxAudioBytes(65537); got != 1 {
		t.Fatalf("MaxAudioBytes(65537)=%d", got)
	}
	want := UpstreamBodyHardMax - MultipartOverheadReserve
	if got := MaxAudioBytes(UpstreamBodyHardMax); got != want {
		t.Fatalf("default 10 MiB audio cap %d", got)
	}
	if got := MaxAudioBytes(UpstreamBodyHardMax + 1); got != want {
		t.Fatalf("above hard max must clamp down, got %d", got)
	}
}

func TestClientHonorsConfiguredBodyLimit(t *testing.T) {
	t.Parallel()
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		hits.Add(1)
	}))
	defer srv.Close()
	audio := bytes.Repeat([]byte("a"), 400)
	c := NewClient(srv.URL, time.Second)
	c.MaxUpstreamBody = 200
	_, err := c.Infer(context.Background(), InferRequest{Audio: audio, Filename: "utt.wav"})
	if !errors.Is(err, ErrTooLarge) {
		t.Fatalf("configured cap ignored: %v", err)
	}
	c.MaxUpstreamBody = 0
	_, err = c.Infer(context.Background(), InferRequest{Audio: []byte("RIFF....WAVE")})
	if !errors.Is(err, ErrTooLarge) {
		t.Fatalf("zero MaxUpstreamBody fail-open: %v", err)
	}
	if hits.Load() != 0 {
		t.Fatalf("upstream called %d times", hits.Load())
	}
}

func okBody() map[string]any {
	return map[string]any{
		"request_id": "11111111-1111-1111-1111-111111111111",
		"model_id":   "c0_cascaded", "asr_text_bahnar": "xin", "translated_text_vi": "tôi",
		"duration_ms": 10, "latency_ms": 7, "tokenizer_parity": "VERIFIED",
		"research_provenance": "NOT_VERIFIED", "checkpoint_identity": "expected_step_match",
		"generation_matches_rq1": true,
	}
}
