//go:build integration

package chat_test

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/chat"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/aiextract"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/config"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/httpserver"
	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
)

// These integration tests exercise the P2 AI-enabled turn path end-to-end
// against a real Postgres database and a fake HTTP server standing in for
// apps/ai-service. They never call a live provider — the fake server here
// plays the exact role apps/ai-service/tests already prove the mock
// provider fulfils.

func newFakeAIServer(t *testing.T, handler func(req aiextract.Request) (aiextract.Response, int)) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req aiextract.Request
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		resp, status := handler(req)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_ = json.NewEncoder(w).Encode(resp)
	}))
	t.Cleanup(srv.Close)
	return srv
}

func testEngineWithAI(t *testing.T, pool *pgxpool.Pool, aiURL string, timeoutMS int) http.Handler {
	t.Helper()
	cfg := config.Config{
		Env:                 "local",
		APIAddr:             ":0",
		DatabaseURL:         pool.Config().ConnString(),
		XAID:                "xa_chu_se",
		LogLevel:            "error",
		CitizenDomainIDs:    []string{"ho_tich_chung_thuc"},
		JWTSecret:           "integration-test-jwt-secret-key",
		JWTExpireHours:      24,
		DBMaxConns:          5,
		DBMinConns:          1,
		AIExtractEnabled:    true,
		AIServiceURL:        aiURL,
		AIExtractTimeoutMS:  timeoutMS,
		AIIntentSelectMin:   0.82,
		AIIntentConfirmMin:  0.55,
		AISlotConfidenceMin: 0.6,
	}
	return httpserver.New(slog.Default(), pool, cfg)
}

func createGuestSession(t *testing.T, eng http.Handler) (sessionID, guest string) {
	t.Helper()
	code, env := apiJSON(t, eng, http.MethodPost, "/api/v1/sessions", map[string]any{"xa_id": "xa_chu_se"}, nil)
	if code != http.StatusCreated {
		t.Fatalf("create session: %d %v", code, env)
	}
	d := dataMap(env)
	sessionID, _ = d["id"].(string)
	guest, _ = d["guest_token"].(string)
	if sessionID == "" || guest == "" {
		t.Fatalf("missing session fields: %v", d)
	}
	return sessionID, guest
}

func lastAssistantMetadata(t *testing.T, eng http.Handler, sessionID, guest string) map[string]any {
	t.Helper()
	code, env := apiJSON(t, eng, http.MethodGet, "/api/v1/sessions/"+sessionID+"/messages", nil,
		map[string]string{"X-Guest-Token": guest})
	if code != http.StatusOK {
		t.Fatalf("list messages: %d %v", code, env)
	}
	items, _ := dataMap(env)["items"].([]any)
	for i := len(items) - 1; i >= 0; i-- {
		m, _ := items[i].(map[string]any)
		if fmt.Sprint(m["role"]) == "ASSISTANT" {
			meta, _ := m["metadata"].(map[string]any)
			return meta
		}
	}
	t.Fatal("no assistant message found")
	return nil
}

func TestHTTPAIExtractFillsSlotEndToEnd(t *testing.T) {
	pool := requireIntegrationDB(t)
	defer pool.Close()

	fake := newFakeAIServer(t, func(req aiextract.Request) (aiextract.Response, int) {
		if req.PinnedContext == nil {
			code := "dk_khai_sinh"
			return aiextract.Response{
				SchemaVersion: aiextract.SchemaVersion,
				Intent:        &aiextract.IntentResult{ProcedureCode: &code, Confidence: 0.95},
				Provider:      "mock", Model: "fake-1",
			}, http.StatusOK
		}
		if req.PinnedContext.ProcedureCode == "dk_khai_sinh" {
			target := "dk_khai_sinh"
			return aiextract.Response{
				SchemaVersion:         aiextract.SchemaVersion,
				SlotsForProcedureCode: &target,
				Slots: []aiextract.SlotResult{
					{Key: "da_ket_hon", Value: false, Confidence: 0.9, Operation: "set"},
				},
				Provider: "mock", Model: "fake-1",
			}, http.StatusOK
		}
		return aiextract.Response{SchemaVersion: aiextract.SchemaVersion, Provider: "mock", Model: "fake-1"}, http.StatusOK
	})
	eng := testEngineWithAI(t, pool, fake.URL, 2000)

	sessionID, guest := createGuestSession(t, eng)

	code, env := apiJSON(t, eng, http.MethodPost, "/api/v1/sessions/"+sessionID+"/turns",
		map[string]any{"message": "Tôi muốn đăng ký khai sinh cho con"},
		map[string]string{"X-Guest-Token": guest, "X-Request-ID": uuid.NewString()})
	if code != http.StatusOK {
		t.Fatalf("turn1: %d %v", code, env)
	}
	d1 := dataMap(env)
	if d1["procedure_code"] != "dk_khai_sinh" {
		t.Fatalf("expected AI to pin dk_khai_sinh, got %v", d1)
	}

	code, env = apiJSON(t, eng, http.MethodPost, "/api/v1/sessions/"+sessionID+"/turns",
		map[string]any{"message": "chưa đăng ký kết hôn"},
		map[string]string{"X-Guest-Token": guest, "X-Request-ID": uuid.NewString()})
	if code != http.StatusOK {
		t.Fatalf("turn2: %d %v", code, env)
	}
	d2 := dataMap(env)
	slotState, _ := d2["slot_state"].(map[string]any)
	daKetHon, _ := slotState["da_ket_hon"].(map[string]any)
	if daKetHon["value"] != false {
		t.Fatalf("expected da_ket_hon=false from AI fill, got %v", slotState)
	}
	if d2["action"] != "ASK_MISSING_SLOTS" {
		t.Fatalf("expected still ASK_MISSING_SLOTS (noi_sinh/co_giay_chung_sinh remain), got %v", d2["action"])
	}

	meta := lastAssistantMetadata(t, eng, sessionID, guest)
	if meta["extract_source"] != "ai" {
		t.Fatalf("expected extract_source=ai in assistant metadata, got %v", meta)
	}
}

func TestHTTPAINoDBLockDuringExtractCall(t *testing.T) {
	pool := requireIntegrationDB(t)
	defer pool.Close()

	called := make(chan struct{}, 1)
	release := make(chan struct{})
	fake := newFakeAIServer(t, func(req aiextract.Request) (aiextract.Response, int) {
		select {
		case called <- struct{}{}:
		default:
		}
		<-release
		code := "dk_khai_sinh"
		return aiextract.Response{
			SchemaVersion: aiextract.SchemaVersion,
			Intent:        &aiextract.IntentResult{ProcedureCode: &code, Confidence: 0.95},
			Provider:      "mock", Model: "fake-1",
		}, http.StatusOK
	})
	eng := testEngineWithAI(t, pool, fake.URL, 5000)
	sessionID, guest := createGuestSession(t, eng)

	done := make(chan struct {
		code int
		env  map[string]any
	}, 1)
	go func() {
		code, env := apiJSON(t, eng, http.MethodPost, "/api/v1/sessions/"+sessionID+"/turns",
			map[string]any{"message": "Tôi muốn đăng ký khai sinh cho con"},
			map[string]string{"X-Guest-Token": guest, "X-Request-ID": uuid.NewString()})
		done <- struct {
			code int
			env  map[string]any
		}{code, env}
	}()

	select {
	case <-called:
	case <-time.After(5 * time.Second):
		close(release)
		t.Fatal("AI server was never called")
	}

	// While the AI call is in flight (blocked), the session row lock from
	// Phase 1 MUST already be released — prove it by acquiring FOR UPDATE
	// NOWAIT on a separate connection. Any lock-held error here means the
	// AI network call happened while still holding the DB transaction.
	sid, err := uuid.Parse(sessionID)
	if err != nil {
		close(release)
		t.Fatal(err)
	}
	checkCtx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	tx, err := pool.Begin(checkCtx)
	if err != nil {
		close(release)
		t.Fatalf("begin check tx: %v", err)
	}
	_, lockErr := tx.Exec(checkCtx, `SELECT 1 FROM conversation_sessions WHERE id=$1 FOR UPDATE NOWAIT`, sid)
	tx.Rollback(checkCtx)
	close(release)
	if lockErr != nil {
		t.Fatalf("expected no DB lock held during AI call, got: %v", lockErr)
	}

	select {
	case result := <-done:
		if result.code != http.StatusOK {
			t.Fatalf("turn after unblocking AI: %d %v", result.code, result.env)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("turn goroutine did not finish after releasing the fake AI server")
	}
}

func TestHTTPAIReplayDoesNotCallExtractorTwice(t *testing.T) {
	pool := requireIntegrationDB(t)
	defer pool.Close()

	var calls int64
	fake := newFakeAIServer(t, func(req aiextract.Request) (aiextract.Response, int) {
		atomic.AddInt64(&calls, 1)
		code := "dk_khai_sinh"
		return aiextract.Response{
			SchemaVersion: aiextract.SchemaVersion,
			Intent:        &aiextract.IntentResult{ProcedureCode: &code, Confidence: 0.95},
			Provider:      "mock", Model: "fake-1",
		}, http.StatusOK
	})
	eng := testEngineWithAI(t, pool, fake.URL, 2000)
	sessionID, guest := createGuestSession(t, eng)

	reqID := uuid.NewString()
	message := "Tôi muốn đăng ký khai sinh cho con"
	code, env := apiJSON(t, eng, http.MethodPost, "/api/v1/sessions/"+sessionID+"/turns",
		map[string]any{"message": message},
		map[string]string{"X-Guest-Token": guest, "X-Request-ID": reqID})
	if code != http.StatusOK {
		t.Fatalf("turn1: %d %v", code, env)
	}
	first := dataMap(env)

	code, env = apiJSON(t, eng, http.MethodPost, "/api/v1/sessions/"+sessionID+"/turns",
		map[string]any{"message": message},
		map[string]string{"X-Guest-Token": guest, "X-Request-ID": reqID})
	if code != http.StatusOK {
		t.Fatalf("replay: %d %v", code, env)
	}
	replay := dataMap(env)
	if replay["idempotent_replay"] != true {
		t.Fatalf("expected idempotent_replay=true, got %v", replay)
	}
	if fmt.Sprint(first["procedure_code"]) != fmt.Sprint(replay["procedure_code"]) {
		t.Fatalf("replay diverged from first result: %v vs %v", first, replay)
	}
	if got := atomic.LoadInt64(&calls); got != 1 {
		t.Fatalf("extractor called %d times, want exactly 1 (replay must not call AI again)", got)
	}
}

func TestHTTPAITimeoutFallsBackToKeywordAndStillSucceeds(t *testing.T) {
	pool := requireIntegrationDB(t)
	defer pool.Close()

	fake := newFakeAIServer(t, func(req aiextract.Request) (aiextract.Response, int) {
		time.Sleep(300 * time.Millisecond)
		code := "dk_khai_sinh"
		return aiextract.Response{
			SchemaVersion: aiextract.SchemaVersion,
			Intent:        &aiextract.IntentResult{ProcedureCode: &code, Confidence: 0.95},
			Provider:      "mock", Model: "fake-1",
		}, http.StatusOK
	})
	eng := testEngineWithAI(t, pool, fake.URL, 30) // 30ms << 300ms server delay
	sessionID, guest := createGuestSession(t, eng)

	code, env := apiJSON(t, eng, http.MethodPost, "/api/v1/sessions/"+sessionID+"/turns",
		map[string]any{"message": "Tôi muốn đăng ký khai sinh cho con"},
		map[string]string{"X-Guest-Token": guest, "X-Request-ID": uuid.NewString()})
	if code != http.StatusOK {
		t.Fatalf("turn should still succeed via keyword fallback: %d %v", code, env)
	}

	meta := lastAssistantMetadata(t, eng, sessionID, guest)
	if meta["extract_source"] != "keyword_fallback" {
		t.Fatalf("expected extract_source=keyword_fallback on timeout, got %v", meta)
	}
	if meta["extract_fallback_reason"] != "timeout" {
		t.Fatalf("expected extract_fallback_reason=timeout, got %v", meta)
	}
}

func TestHTTPAIInvalidSlotFallsBackToKeyword(t *testing.T) {
	pool := requireIntegrationDB(t)
	defer pool.Close()

	fake := newFakeAIServer(t, func(req aiextract.Request) (aiextract.Response, int) {
		if req.PinnedContext == nil {
			code := "dk_khai_sinh"
			return aiextract.Response{
				SchemaVersion: aiextract.SchemaVersion,
				Intent:        &aiextract.IntentResult{ProcedureCode: &code, Confidence: 0.95},
				Provider:      "mock", Model: "fake-1",
			}, http.StatusOK
		}
		target := req.PinnedContext.ProcedureCode
		return aiextract.Response{
			SchemaVersion:         aiextract.SchemaVersion,
			SlotsForProcedureCode: &target,
			Slots: []aiextract.SlotResult{
				// Not a real slot on this procedure at all — must be rejected
				// by aiextract.Validate and trigger a full-response fallback.
				{Key: "totally_unknown_key", Value: "x", Confidence: 0.9, Operation: "set"},
			},
			Provider: "mock", Model: "fake-1",
		}, http.StatusOK
	})
	eng := testEngineWithAI(t, pool, fake.URL, 2000)
	sessionID, guest := createGuestSession(t, eng)

	code, env := apiJSON(t, eng, http.MethodPost, "/api/v1/sessions/"+sessionID+"/turns",
		map[string]any{"message": "Tôi muốn đăng ký khai sinh cho con"},
		map[string]string{"X-Guest-Token": guest, "X-Request-ID": uuid.NewString()})
	if code != http.StatusOK {
		t.Fatalf("turn1: %d %v", code, env)
	}

	code, env = apiJSON(t, eng, http.MethodPost, "/api/v1/sessions/"+sessionID+"/turns",
		map[string]any{"message": "một câu trả lời bất kỳ"},
		map[string]string{"X-Guest-Token": guest, "X-Request-ID": uuid.NewString()})
	if code != http.StatusOK {
		t.Fatalf("turn2 should still succeed via fallback, not 500: %d %v", code, env)
	}
	d2 := dataMap(env)
	slotState, _ := d2["slot_state"].(map[string]any)
	if _, ok := slotState["totally_unknown_key"]; ok {
		t.Fatalf("unknown slot key must never reach persisted slot state: %v", slotState)
	}

	meta := lastAssistantMetadata(t, eng, sessionID, guest)
	if meta["extract_fallback_reason"] != "invalid_contract" {
		t.Fatalf("expected extract_fallback_reason=invalid_contract, got %v", meta)
	}
}

func TestHTTPAISameRequestIDDifferentBodyStill409(t *testing.T) {
	pool := requireIntegrationDB(t)
	defer pool.Close()

	fake := newFakeAIServer(t, func(req aiextract.Request) (aiextract.Response, int) {
		return aiextract.Response{SchemaVersion: aiextract.SchemaVersion, Provider: "mock", Model: "fake-1"}, http.StatusOK
	})
	eng := testEngineWithAI(t, pool, fake.URL, 2000)
	sessionID, guest := createGuestSession(t, eng)

	reqID := uuid.NewString()
	code, env := apiJSON(t, eng, http.MethodPost, "/api/v1/sessions/"+sessionID+"/turns",
		map[string]any{"message": "Tôi muốn đăng ký khai sinh cho con"},
		map[string]string{"X-Guest-Token": guest, "X-Request-ID": reqID})
	if code != http.StatusOK {
		t.Fatalf("turn1: %d %v", code, env)
	}

	code, env = apiJSON(t, eng, http.MethodPost, "/api/v1/sessions/"+sessionID+"/turns",
		map[string]any{"message": "một câu hoàn toàn khác"},
		map[string]string{"X-Guest-Token": guest, "X-Request-ID": reqID})
	if code != http.StatusConflict {
		t.Fatalf("want 409 IDEMPOTENCY_CONFLICT, got %d %v", code, env)
	}
	if errObj, _ := env["error"].(map[string]any); fmt.Sprint(errObj["code"]) != "IDEMPOTENCY_CONFLICT" {
		t.Fatalf("want IDEMPOTENCY_CONFLICT, got %v", env)
	}
}

func TestHTTPAIConcurrentSameRequestCallsExtractorOnce(t *testing.T) {
	pool := requireIntegrationDB(t)
	defer pool.Close()

	called := make(chan struct{}, 1)
	release := make(chan struct{})
	var calls int64
	fake := newFakeAIServer(t, func(req aiextract.Request) (aiextract.Response, int) {
		n := atomic.AddInt64(&calls, 1)
		if n == 1 {
			called <- struct{}{}
			<-release
		}
		code := "dk_khai_sinh"
		return aiextract.Response{
			SchemaVersion: aiextract.SchemaVersion,
			Intent:        &aiextract.IntentResult{ProcedureCode: &code, Confidence: 0.95},
			Provider:      "mock", Model: "fake-1",
		}, http.StatusOK
	})
	eng := testEngineWithAI(t, pool, fake.URL, 5000)
	sessionID, guest := createGuestSession(t, eng)
	reqID := uuid.NewString()
	message := "Tôi muốn đăng ký khai sinh cho con"
	path := "/api/v1/sessions/" + sessionID + "/turns"
	headers := map[string]string{"X-Guest-Token": guest, "X-Request-ID": reqID}

	type result struct {
		code int
		env  map[string]any
	}
	var wg sync.WaitGroup
	out := make(chan result, 2)
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			code, env := apiJSON(t, eng, http.MethodPost, path, map[string]any{"message": message}, headers)
			out <- result{code, env}
		}()
	}

	select {
	case <-called:
	case <-time.After(5 * time.Second):
		close(release)
		t.Fatal("extractor was not called")
	}

	code, env := apiJSON(t, eng, http.MethodPost, path,
		map[string]any{"message": "một câu hoàn toàn khác"}, headers)
	if code != http.StatusConflict {
		close(release)
		t.Fatalf("different body during claim: want 409, got %d %v", code, env)
	}
	if atomic.LoadInt64(&calls) != 1 {
		close(release)
		t.Fatalf("different body must not call the extractor, calls=%d", calls)
	}

	close(release)
	wg.Wait()
	close(out)
	var first map[string]any
	for res := range out {
		if res.code != http.StatusOK {
			t.Fatalf("concurrent turn: %d %v", res.code, res.env)
		}
		d := dataMap(res.env)
		if first == nil {
			first = d
			continue
		}
		if fmt.Sprint(first["procedure_code"]) != fmt.Sprint(d["procedure_code"]) ||
			fmt.Sprint(first["action"]) != fmt.Sprint(d["action"]) {
			t.Fatalf("concurrent results diverged: %v vs %v", first, d)
		}
	}
	if got := atomic.LoadInt64(&calls); got != 1 {
		t.Fatalf("extractor called %d times, want 1", got)
	}
}

func TestHTTPAIStalePendingClaimIsRecovered(t *testing.T) {
	pool := requireIntegrationDB(t)
	defer pool.Close()

	var calls int64
	fake := newFakeAIServer(t, func(req aiextract.Request) (aiextract.Response, int) {
		atomic.AddInt64(&calls, 1)
		code := "dk_khai_sinh"
		return aiextract.Response{
			SchemaVersion: aiextract.SchemaVersion,
			Intent:        &aiextract.IntentResult{ProcedureCode: &code, Confidence: 0.95},
			Provider:      "mock", Model: "fake-1",
		}, http.StatusOK
	})
	eng := testEngineWithAI(t, pool, fake.URL, 2000)
	sessionID, guest := createGuestSession(t, eng)
	reqID := uuid.NewString()
	message := "Tôi muốn đăng ký khai sinh cho con"
	hash := chat.PayloadHash(message)

	ctx := context.Background()
	_, err := pool.Exec(ctx, `
		INSERT INTO turn_idempotency
			(session_id, request_id, payload_hash, response_json, status, claimed_at, claim_expires_at)
		VALUES ($1, $2, $3, NULL, 'PENDING', now() - interval '2 minutes', now() - interval '1 minute')`,
		sessionID, reqID, hash)
	if err != nil {
		t.Fatalf("insert stale claim: %v", err)
	}

	code, env := apiJSON(t, eng, http.MethodPost, "/api/v1/sessions/"+sessionID+"/turns",
		map[string]any{"message": message},
		map[string]string{"X-Guest-Token": guest, "X-Request-ID": reqID})
	if code != http.StatusOK {
		t.Fatalf("recovered turn: %d %v", code, env)
	}
	if atomic.LoadInt64(&calls) != 1 {
		t.Fatalf("stale claim should call the extractor once, got %d", calls)
	}
	var status string
	if err := pool.QueryRow(ctx, `SELECT status FROM turn_idempotency WHERE session_id=$1 AND request_id=$2`, sessionID, reqID).Scan(&status); err != nil {
		t.Fatal(err)
	}
	if status != "COMPLETE" {
		t.Fatalf("status = %s, want COMPLETE", status)
	}
}

func TestHTTPAIRejectsSpoofedProviderInMetadata(t *testing.T) {
	pool := requireIntegrationDB(t)
	defer pool.Close()

	const pii = "PII-TOKEN-7788"
	fake := newFakeAIServer(t, func(req aiextract.Request) (aiextract.Response, int) {
		return aiextract.Response{
			SchemaVersion: aiextract.SchemaVersion,
			Provider:      pii,
			Model:         "Nguyen Van A 0901234567",
		}, http.StatusOK
	})
	eng := testEngineWithAI(t, pool, fake.URL, 2000)
	sessionID, guest := createGuestSession(t, eng)
	code, env := apiJSON(t, eng, http.MethodPost, "/api/v1/sessions/"+sessionID+"/turns",
		map[string]any{"message": "xin chào"},
		map[string]string{"X-Guest-Token": guest, "X-Request-ID": uuid.NewString()})
	if code != http.StatusOK {
		t.Fatalf("turn: %d %v", code, env)
	}
	meta := lastAssistantMetadata(t, eng, sessionID, guest)
	raw, _ := json.Marshal(meta)
	if strings.Contains(string(raw), pii) || strings.Contains(string(raw), "0901234567") {
		t.Fatalf("metadata stored model-controlled text: %s", raw)
	}
	if meta["extract_fallback_reason"] != "invalid_contract" {
		t.Fatalf("fallback = %v", meta["extract_fallback_reason"])
	}
	if _, ok := meta["extract_provider"]; ok {
		t.Fatalf("spoofed provider was stored: %v", meta["extract_provider"])
	}
}

func TestHTTPAI429FallsBackWithoutRetry(t *testing.T) {
	pool := requireIntegrationDB(t)
	defer pool.Close()

	var calls int64
	fake := newFakeAIServer(t, func(req aiextract.Request) (aiextract.Response, int) {
		atomic.AddInt64(&calls, 1)
		return aiextract.Response{}, http.StatusTooManyRequests
	})
	eng := testEngineWithAI(t, pool, fake.URL, 2000)
	sessionID, guest := createGuestSession(t, eng)
	code, env := apiJSON(t, eng, http.MethodPost, "/api/v1/sessions/"+sessionID+"/turns",
		map[string]any{"message": "xin chào"},
		map[string]string{"X-Guest-Token": guest, "X-Request-ID": uuid.NewString()})
	if code != http.StatusOK {
		t.Fatalf("429 must fall back to keyword, got %d %v", code, env)
	}
	if atomic.LoadInt64(&calls) != 1 {
		t.Fatalf("extractor calls = %d, want 1", calls)
	}
	meta := lastAssistantMetadata(t, eng, sessionID, guest)
	if meta["extract_fallback_reason"] != "rate_limited" {
		t.Fatalf("fallback = %v", meta["extract_fallback_reason"])
	}
}
