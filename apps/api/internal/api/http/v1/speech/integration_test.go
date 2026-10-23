package speechapi

import (
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/speech"
	"github.com/gin-gonic/gin"
)

func TestGinRouteToFakeSpeechAPI(t *testing.T) {
	gin.SetMode(gin.TestMode)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != speech.InferPath {
			http.NotFound(w, r)
			return
		}
		if r.Header.Get("Authorization") != "Bearer speech-service-token-1" {
			w.WriteHeader(http.StatusUnauthorized)
			_, _ = io.WriteString(w, `{"error":{"code":"unauthorized"}}`)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"request_id": r.Header.Get("X-Request-ID"), "model_id": "c0_cascaded",
			"asr_text_bahnar": "xin", "translated_text_vi": "tôi",
			"duration_ms": 14432, "latency_ms": 529,
			"tokenizer_parity": "VERIFIED", "research_provenance": "NOT_VERIFIED",
			"checkpoint_identity": "expected_step_match", "generation_matches_rq1": true,
		})
	}))
	defer upstream.Close()

	sess := guestSession()
	audit := &fakeAudit{}
	client := speech.NewClient(upstream.URL, time.Second)
	client.ServiceToken = "speech-service-token-1"
	h := NewHandler(Handler{
		Enabled: true, Client: client, Sessions: sessionsOf(sess),
		Audit: audit, MaxUploadBytes: 1024, MaxInflight: 1,
		Logger: slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	res := postTranslate(t, h, sess, wavBytes(), map[string]string{HeaderGuestToken: *sess.GuestToken})
	if res.Code != http.StatusOK {
		t.Fatalf("%d %s", res.Code, res.Body.String())
	}
	if len(audit.recs) != 1 || audit.recs[0].Outcome != speech.OutcomeSuccess {
		t.Fatalf("audit %+v", audit.recs)
	}
	if audit.recs[0].ResearchProvenance != "NOT_VERIFIED" {
		t.Fatal("invented verification")
	}
}

func TestDisabledConfigMakesNoUpstreamCall(t *testing.T) {
	gin.SetMode(gin.TestMode)
	var hits int
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits++
		w.WriteHeader(http.StatusOK)
	}))
	defer upstream.Close()
	sess := guestSession()
	h := NewHandler(Handler{
		Enabled: false, Client: speech.NewClient(upstream.URL, time.Second),
		Sessions: sessionsOf(sess), Audit: &fakeAudit{},
		MaxUploadBytes: 1024, Logger: slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	res := postTranslate(t, h, sess, wavBytes(), map[string]string{HeaderGuestToken: *sess.GuestToken})
	assertCode(t, res, http.StatusServiceUnavailable, "SPEECH_DISABLED")
	if hits != 0 {
		t.Fatalf("hits %d", hits)
	}
}
