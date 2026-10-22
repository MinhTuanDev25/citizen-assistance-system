//go:build integration

package chat_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/auth"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/chat"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/config"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/httpserver"
	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
)

func requireIntegrationDB(t *testing.T) *pgxpool.Pool {
	t.Helper()
	if os.Getenv("CAS_INTEGRATION") != "1" && os.Getenv("CAS_INTEGRATION") != "true" {
		t.Fatal("CAS_INTEGRATION=1 required: integration tests must not silently skip")
	}
	url := os.Getenv("DATABASE_URL")
	if url == "" {
		url = "postgres://cas:cas@127.0.0.1:5432/citizen_assistance?sslmode=disable"
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	pool, err := pgxpool.New(ctx, url)
	if err != nil {
		t.Fatalf("DATABASE_URL required for integration: %v", err)
	}
	if err := pool.Ping(ctx); err != nil {
		t.Fatalf("postgres unavailable (integration must fail, not skip): %v", err)
	}
	return pool
}

func testEngine(t *testing.T, pool *pgxpool.Pool) http.Handler {
	t.Helper()
	cfg := config.Config{
		Env:              "local",
		APIAddr:          ":0",
		DatabaseURL:      pool.Config().ConnString(),
		XAID:             "xa_chu_se",
		LogLevel:         "error",
		CitizenDomainIDs: []string{"ho_tich_chung_thuc"},
		JWTSecret:        "integration-test-jwt-secret-key",
		JWTExpireHours:   24,
		DBMaxConns:       5,
		DBMinConns:       1,
	}
	return httpserver.New(slog.Default(), pool, cfg)
}

func apiJSON(t *testing.T, eng http.Handler, method, path string, body any, headers map[string]string) (int, map[string]any) {
	t.Helper()
	var buf bytes.Buffer
	if body != nil {
		if err := json.NewEncoder(&buf).Encode(body); err != nil {
			t.Fatal(err)
		}
	}
	req := httptest.NewRequest(method, path, &buf)
	req.Header.Set("Content-Type", "application/json")
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	w := httptest.NewRecorder()
	eng.ServeHTTP(w, req)
	var env map[string]any
	_ = json.Unmarshal(w.Body.Bytes(), &env)
	return w.Code, env
}

func dataMap(env map[string]any) map[string]any {
	d, _ := env["data"].(map[string]any)
	return d
}

func assertTurnEnvelopeEqual(t *testing.T, a, b map[string]any) {
	t.Helper()
	keys := []string{"action", "reply_text", "procedure_code", "procedure_version"}
	for _, k := range keys {
		if fmt.Sprint(a[k]) != fmt.Sprint(b[k]) {
			t.Fatalf("%s mismatch: %v vs %v", k, a[k], b[k])
		}
	}
	for _, k := range []string{"ask_now", "questions", "guidance", "candidates", "citations", "slot_state", "missing_slots", "filled_slots"} {
		aj, _ := json.Marshal(a[k])
		bj, _ := json.Marshal(b[k])
		if string(aj) != string(bj) {
			t.Fatalf("%s mismatch: %s vs %s", k, aj, bj)
		}
	}
	umA, _ := a["user_message"].(map[string]any)
	umB, _ := b["user_message"].(map[string]any)
	amA, _ := a["assistant_message"].(map[string]any)
	amB, _ := b["assistant_message"].(map[string]any)
	if fmt.Sprint(umA["id"]) != fmt.Sprint(umB["id"]) || fmt.Sprint(amA["id"]) != fmt.Sprint(amB["id"]) {
		t.Fatalf("message id mismatch user %v/%v asst %v/%v", umA["id"], umB["id"], amA["id"], amB["id"])
	}
}

func TestHTTPGuestIsolationAndTurns(t *testing.T) {
	pool := requireIntegrationDB(t)
	defer pool.Close()
	eng := testEngine(t, pool)

	code, env := apiJSON(t, eng, http.MethodPost, "/api/v1/sessions", map[string]any{"xa_id": "xa_chu_se"}, nil)
	if code != http.StatusCreated {
		t.Fatalf("create session %d %v", code, env)
	}
	d := dataMap(env)
	sessionID, _ := d["id"].(string)
	guest, _ := d["guest_token"].(string)
	if sessionID == "" || guest == "" {
		t.Fatalf("missing session fields: %v", d)
	}

	code, env = apiJSON(t, eng, http.MethodPost, "/api/v1/sessions/"+sessionID+"/turns",
		map[string]any{"message": "Đăng ký khai sinh"},
		map[string]string{"X-Guest-Token": "wrong-token", "X-Request-ID": uuid.NewString()})
	if code != http.StatusForbidden {
		t.Fatalf("want 403 got %d %v", code, env)
	}

	code, _ = apiJSON(t, eng, http.MethodPost, "/api/v1/sessions/"+sessionID+"/turns",
		map[string]any{"message": "Đăng ký khai sinh"},
		map[string]string{"X-Request-ID": uuid.NewString()})
	if code != http.StatusUnauthorized {
		t.Fatalf("want 401 got %d", code)
	}

	// Invalid X-Request-ID → 400 (must not mint a silent substitute)
	code, env = apiJSON(t, eng, http.MethodPost, "/api/v1/sessions/"+sessionID+"/turns",
		map[string]any{"message": "Đăng ký khai sinh"},
		map[string]string{"X-Guest-Token": guest, "X-Request-ID": "not-a-uuid"})
	if code != http.StatusBadRequest {
		t.Fatalf("invalid request id want 400 got %d %v", code, env)
	}
	code, env = apiJSON(t, eng, http.MethodPost, "/api/v1/sessions/"+sessionID+"/turns",
		map[string]any{"message": "Đăng ký khai sinh"},
		map[string]string{"X-Guest-Token": guest})
	if code != http.StatusBadRequest {
		t.Fatalf("missing request id want 400 got %d %v", code, env)
	}

	reqID := uuid.NewString()
	code, env = apiJSON(t, eng, http.MethodPost, "/api/v1/sessions/"+sessionID+"/turns",
		map[string]any{"message": "Đăng ký khai sinh"},
		map[string]string{"X-Guest-Token": guest, "X-Request-ID": reqID})
	if code != http.StatusOK {
		t.Fatalf("turn %d %v", code, env)
	}
	first := dataMap(env)
	if first["action"] == nil {
		t.Fatalf("missing action: %v", first)
	}

	code, env = apiJSON(t, eng, http.MethodPost, "/api/v1/sessions/"+sessionID+"/turns",
		map[string]any{"message": "Đăng ký khai sinh"},
		map[string]string{"X-Guest-Token": guest, "X-Request-ID": reqID})
	if code != http.StatusOK {
		t.Fatalf("replay %d %v", code, env)
	}
	replay := dataMap(env)
	if replay["idempotent_replay"] != true {
		t.Fatalf("expected idempotent_replay: %v", replay)
	}
	assertTurnEnvelopeEqual(t, first, replay)

	code, env = apiJSON(t, eng, http.MethodPost, "/api/v1/sessions/"+sessionID+"/turns",
		map[string]any{"message": "Chứng thực bản sao"},
		map[string]string{"X-Guest-Token": guest, "X-Request-ID": reqID})
	if code != http.StatusConflict {
		t.Fatalf("want 409 conflict got %d %v", code, env)
	}
	if errObj, _ := env["error"].(map[string]any); fmt.Sprint(errObj["code"]) != "IDEMPOTENCY_CONFLICT" {
		t.Fatalf("want IDEMPOTENCY_CONFLICT got %v", env)
	}

	code, env = apiJSON(t, eng, http.MethodPost, "/api/v1/sessions/"+sessionID+"/messages",
		map[string]any{"message": "should fail"},
		map[string]string{"X-Guest-Token": guest})
	if code != http.StatusGone {
		t.Fatalf("want 410 got %d %v", code, env)
	}

	code, env = apiJSON(t, eng, http.MethodPost, "/api/v1/sessions", map[string]any{"xa_id": "xa_chu_se"}, nil)
	otherGuest, _ := dataMap(env)["guest_token"].(string)
	code, env = apiJSON(t, eng, http.MethodGet, "/api/v1/sessions/"+sessionID+"/messages", nil,
		map[string]string{"X-Guest-Token": otherGuest})
	if code != http.StatusForbidden {
		t.Fatalf("cross-guest history want 403 got %d %v", code, env)
	}

	_, _ = pool.Exec(context.Background(), `UPDATE conversation_sessions SET status='COMPLETED' WHERE id=$1`, sessionID)
	code, env = apiJSON(t, eng, http.MethodPost, "/api/v1/sessions/"+sessionID+"/turns",
		map[string]any{"message": "hello"},
		map[string]string{"X-Guest-Token": guest, "X-Request-ID": uuid.NewString()})
	if code != http.StatusConflict {
		t.Fatalf("closed session want 409 got %d %v", code, env)
	}
}

func TestHTTPLegacyIdempotencyWithoutEnvelope(t *testing.T) {
	pool := requireIntegrationDB(t)
	defer pool.Close()
	eng := testEngine(t, pool)
	ctx := context.Background()

	_, env := apiJSON(t, eng, http.MethodPost, "/api/v1/sessions", map[string]any{"xa_id": "xa_chu_se"}, nil)
	d := dataMap(env)
	sessionID, _ := d["id"].(string)
	guest, _ := d["guest_token"].(string)
	reqID := uuid.NewString()

	code, env := apiJSON(t, eng, http.MethodPost, "/api/v1/sessions/"+sessionID+"/turns",
		map[string]any{"message": "Đăng ký khai sinh"},
		map[string]string{"X-Guest-Token": guest, "X-Request-ID": reqID})
	if code != http.StatusOK {
		t.Fatalf("setup turn %d %v", code, env)
	}
	first := dataMap(env)

	tag, err := pool.Exec(ctx, `DELETE FROM turn_idempotency WHERE session_id=$1 AND request_id=$2`, sessionID, reqID)
	if err != nil {
		t.Fatal(err)
	}
	if tag.RowsAffected() < 1 {
		t.Fatal("expected turn_idempotency row to delete")
	}

	code, env = apiJSON(t, eng, http.MethodPost, "/api/v1/sessions/"+sessionID+"/turns",
		map[string]any{"message": "Đăng ký khai sinh"},
		map[string]string{"X-Guest-Token": guest, "X-Request-ID": reqID})
	if code != http.StatusOK {
		t.Fatalf("legacy replay must not 500: %d %v", code, env)
	}
	replay := dataMap(env)
	if replay["idempotent_replay"] != true {
		t.Fatalf("expected legacy idempotent_replay: %v", replay)
	}
	assertTurnEnvelopeEqual(t, first, replay)
}

func TestHTTPAuthIsolation(t *testing.T) {
	pool := requireIntegrationDB(t)
	defer pool.Close()
	eng := testEngine(t, pool)
	ctx := context.Background()

	hashA, err := auth.HashPassword("password-a-long")
	if err != nil {
		t.Fatal(err)
	}
	hashB, err := auth.HashPassword("password-b-long")
	if err != nil {
		t.Fatal(err)
	}
	idA := uuid.New()
	idB := uuid.New()
	_, err = pool.Exec(ctx, `
		INSERT INTO users (id, role, full_name, email, password_hash)
		VALUES ($1,'CITIZEN','User A',$2,$3), ($4,'CITIZEN','User B',$5,$6)`,
		idA, "user-a-"+idA.String()[:8]+"@example.com", hashA,
		idB, "user-b-"+idB.String()[:8]+"@example.com", hashB)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_, _ = pool.Exec(ctx, `DELETE FROM conversation_messages WHERE session_id IN (SELECT id FROM conversation_sessions WHERE user_id IN ($1,$2))`, idA, idB)
		_, _ = pool.Exec(ctx, `DELETE FROM turn_idempotency WHERE session_id IN (SELECT id FROM conversation_sessions WHERE user_id IN ($1,$2))`, idA, idB)
		_, _ = pool.Exec(ctx, `DELETE FROM conversation_sessions WHERE user_id IN ($1,$2)`, idA, idB)
		_, _ = pool.Exec(ctx, `DELETE FROM users WHERE id IN ($1,$2)`, idA, idB)
	})

	tokSvc := &auth.TokenService{Secret: []byte("integration-test-jwt-secret-key"), TTL: time.Hour}
	tokA, _, err := tokSvc.Issue(idA, auth.RoleCitizen, "a@example.com", "A")
	if err != nil {
		t.Fatal(err)
	}
	tokB, _, err := tokSvc.Issue(idB, auth.RoleCitizen, "b@example.com", "B")
	if err != nil {
		t.Fatal(err)
	}
	expiredTok, err := issueExpiredToken(idA)
	if err != nil {
		t.Fatal(err)
	}
	_ = expiredTok
	badTok := "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.e30.bad"

	code, env := apiJSON(t, eng, http.MethodPost, "/api/v1/sessions", map[string]any{"xa_id": "xa_chu_se"},
		map[string]string{"Authorization": "Bearer " + tokA})
	if code != http.StatusCreated {
		t.Fatalf("user A session %d %v", code, env)
	}
	sessionA := fmt.Sprint(dataMap(env)["id"])

	code, env = apiJSON(t, eng, http.MethodPost, "/api/v1/sessions", map[string]any{"xa_id": "xa_chu_se"}, nil)
	guestTok, _ := dataMap(env)["guest_token"].(string)
	guestSess := fmt.Sprint(dataMap(env)["id"])

	// User B cannot read User A session
	code, env = apiJSON(t, eng, http.MethodGet, "/api/v1/sessions/"+sessionA+"/messages", nil,
		map[string]string{"Authorization": "Bearer " + tokB})
	if code != http.StatusForbidden {
		t.Fatalf("B read A want 403 got %d %v", code, env)
	}
	code, env = apiJSON(t, eng, http.MethodPost, "/api/v1/sessions/"+sessionA+"/turns",
		map[string]any{"message": "hi"},
		map[string]string{"Authorization": "Bearer " + tokB, "X-Request-ID": uuid.NewString()})
	if code != http.StatusForbidden {
		t.Fatalf("B turn A want 403 got %d %v", code, env)
	}

	// Guest cannot access user session
	code, env = apiJSON(t, eng, http.MethodGet, "/api/v1/sessions/"+sessionA+"/messages", nil,
		map[string]string{"X-Guest-Token": guestTok})
	if code != http.StatusForbidden && code != http.StatusUnauthorized {
		t.Fatalf("guest on user session want 403/401 got %d %v", code, env)
	}

	// User cannot use guest token to claim guest session as owner via turns with JWT of another user
	code, env = apiJSON(t, eng, http.MethodPost, "/api/v1/sessions/"+guestSess+"/turns",
		map[string]any{"message": "Đăng ký khai sinh"},
		map[string]string{"Authorization": "Bearer " + tokA, "X-Request-ID": uuid.NewString()})
	if code != http.StatusUnauthorized && code != http.StatusForbidden {
		t.Fatalf("user JWT on guest session without guest token want 401/403 got %d %v", code, env)
	}

	// Invalid JWT rejected
	code, env = apiJSON(t, eng, http.MethodGet, "/api/v1/auth/me", nil,
		map[string]string{"Authorization": "Bearer " + badTok})
	if code != http.StatusUnauthorized {
		t.Fatalf("bad jwt want 401 got %d %v", code, env)
	}

	// Expired JWT rejected
	code, env = apiJSON(t, eng, http.MethodGet, "/api/v1/auth/me", nil,
		map[string]string{"Authorization": "Bearer " + expiredTok})
	if code != http.StatusUnauthorized {
		t.Fatalf("expired jwt want 401 got %d %v", code, env)
	}
}

func issueExpiredToken(userID uuid.UUID) (string, error) {
	svc := &auth.TokenService{Secret: []byte("integration-test-jwt-secret-key"), TTL: -time.Hour}
	tok, _, err := svc.Issue(userID, auth.RoleCitizen, "expired@example.com", "X")
	return tok, err
}

func TestHTTPConfirmIntentPinsV1AfterActiveV2(t *testing.T) {
	pool := requireIntegrationDB(t)
	defer pool.Close()
	eng := testEngine(t, pool)
	ctx := context.Background()

	// Always pin against the seed 1.0.0 row (not whatever ACTIVE leftover may be).
	var ksID, ctID, ksVer, ctVer uuid.UUID
	var ksDef, ctDef []byte
	if err := pool.QueryRow(ctx, `
		SELECT p.id, pv.id, pv.definition FROM procedures p
		JOIN procedure_versions pv ON pv.procedure_id = p.id AND pv.version = '1.0.0'
		WHERE p.xa_id='xa_chu_se' AND p.procedure_code='dk_khai_sinh'`).Scan(&ksID, &ksVer, &ksDef); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(ctx, `
		SELECT p.id, pv.id, pv.definition FROM procedures p
		JOIN procedure_versions pv ON pv.procedure_id = p.id AND pv.version = '1.0.0'
		WHERE p.xa_id='xa_chu_se' AND p.procedure_code='chung_thuc_ban_sao'`).Scan(&ctID, &ctVer, &ctDef); err != nil {
		t.Fatal(err)
	}
	// Drop leftover v2 rows from prior failed cleanups and force ACTIVE → 1.0.0.
	_, _ = pool.Exec(ctx, `
		UPDATE conversation_sessions
		SET active_procedure_id=NULL, active_procedure_version_id=NULL
		WHERE active_procedure_id IN ($1,$2)`, ksID, ctID)
	_, _ = pool.Exec(ctx, `UPDATE procedures SET active_version_id=$2 WHERE id=$1`, ksID, ksVer)
	_, _ = pool.Exec(ctx, `UPDATE procedures SET active_version_id=$2 WHERE id=$1`, ctID, ctVer)
	_, _ = pool.Exec(ctx, `UPDATE procedure_versions SET status='ARCHIVED' WHERE procedure_id IN ($1,$2) AND id NOT IN ($3,$4)`, ksID, ctID, ksVer, ctVer)
	_, _ = pool.Exec(ctx, `DELETE FROM procedure_versions WHERE procedure_id IN ($1,$2) AND id NOT IN ($3,$4)`, ksID, ctID, ksVer, ctVer)
	_, _ = pool.Exec(ctx, `UPDATE procedure_versions SET status='ACTIVE', definition=$2::jsonb WHERE id=$1`, ksVer, ksDef)
	_, _ = pool.Exec(ctx, `UPDATE procedure_versions SET status='ACTIVE', definition=$2::jsonb WHERE id=$1`, ctVer, ctDef)
	_, _ = pool.Exec(ctx, `UPDATE procedures SET active_version_id=$2 WHERE id=$1`, ksID, ksVer)
	_, _ = pool.Exec(ctx, `UPDATE procedures SET active_version_id=$2 WHERE id=$1`, ctID, ctVer)

	var ksName, ctName string
	_ = pool.QueryRow(ctx, `SELECT name FROM procedures WHERE id=$1`, ksID).Scan(&ksName)
	_ = pool.QueryRow(ctx, `SELECT name FROM procedures WHERE id=$1`, ctID).Scan(&ctName)
	// Neutralize names so shared examples alone drive a mid-band CONFIRM_INTENT (not a name-based select).
	if _, err := pool.Exec(ctx, `UPDATE procedures SET name='Thủ tục mẫu Alpha' WHERE id=$1`, ksID); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `UPDATE procedures SET name='Thủ tục mẫu Beta' WHERE id=$1`, ctID); err != nil {
		t.Fatal(err)
	}

	patchExamples := func(def []byte) []byte {
		var m map[string]any
		if err := json.Unmarshal(def, &m); err != nil {
			t.Fatal(err)
		}
		m["name"] = "Thủ tục mẫu"
		m["intent_examples"] = []any{"đăng ký giấy tờ hành chính xã", "làm giấy tờ đăng ký"}
		out, _ := json.Marshal(m)
		return out
	}
	ksPatched := patchExamples(ksDef)
	ctPatched := patchExamples(ctDef)
	if _, err := pool.Exec(ctx, `UPDATE procedure_versions SET definition=$2::jsonb WHERE id=$1`, ksVer, ksPatched); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `UPDATE procedure_versions SET definition=$2::jsonb WHERE id=$1`, ctVer, ctPatched); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(ctx, `SELECT definition FROM procedure_versions WHERE id=$1`, ksVer).Scan(&ksPatched); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(ctx, `SELECT definition FROM procedure_versions WHERE id=$1`, ctVer).Scan(&ctPatched); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_, _ = pool.Exec(ctx, `UPDATE conversation_sessions SET active_procedure_id=NULL, active_procedure_version_id=NULL WHERE active_procedure_id IN ($1,$2)`, ksID, ctID)
		_, _ = pool.Exec(ctx, `UPDATE procedures SET active_version_id=$2 WHERE id=$1`, ksID, ksVer)
		_, _ = pool.Exec(ctx, `UPDATE procedures SET active_version_id=$2 WHERE id=$1`, ctID, ctVer)
		_, _ = pool.Exec(ctx, `DELETE FROM procedure_versions WHERE procedure_id IN ($1,$2) AND id NOT IN ($3,$4)`, ksID, ctID, ksVer, ctVer)
		_, _ = pool.Exec(ctx, `UPDATE procedure_versions SET status='ACTIVE', definition=$2::jsonb WHERE id=$1`, ksVer, ksDef)
		_, _ = pool.Exec(ctx, `UPDATE procedure_versions SET status='ACTIVE', definition=$2::jsonb WHERE id=$1`, ctVer, ctDef)
		_, _ = pool.Exec(ctx, `UPDATE procedures SET active_version_id=$2, name=$3 WHERE id=$1`, ksID, ksVer, ksName)
		_, _ = pool.Exec(ctx, `UPDATE procedures SET active_version_id=$2, name=$3 WHERE id=$1`, ctID, ctVer, ctName)
	})

	code, catEnv := apiJSON(t, eng, http.MethodGet, "/api/v1/procedures?xa_id=xa_chu_se&citizen=true", nil, nil)
	if code != http.StatusOK {
		t.Fatalf("catalog setup %d %v", code, catEnv)
	}
	if int(dataMap(catEnv)["count"].(float64)) < 2 {
		t.Fatalf("citizen catalog must include both seed procedures before confirm, got %v", dataMap(catEnv))
	}

	_, env := apiJSON(t, eng, http.MethodPost, "/api/v1/sessions", map[string]any{"xa_id": "xa_chu_se"}, nil)
	d := dataMap(env)
	sessionID, _ := d["id"].(string)
	guest, _ := d["guest_token"].(string)

	code, turnEnv := apiJSON(t, eng, http.MethodPost, "/api/v1/sessions/"+sessionID+"/turns",
		map[string]any{"message": "đăng ký giấy tờ hành chính xã"},
		map[string]string{"X-Guest-Token": guest, "X-Request-ID": uuid.NewString()})
	if code != http.StatusOK {
		t.Fatalf("confirm turn %d %v", code, turnEnv)
	}
	td := dataMap(turnEnv)
	if fmt.Sprint(td["action"]) != "CONFIRM_INTENT" {
		t.Fatalf("want CONFIRM_INTENT got %v", td)
	}
	cands, _ := td["candidates"].([]any)
	if len(cands) < 1 {
		t.Fatalf("need candidates: %v", td)
	}
	c0, _ := cands[0].(map[string]any)
	pinnedVerID := fmt.Sprint(c0["procedure_version_id"])
	pinnedHash := fmt.Sprint(c0["definition_hash"])
	pinnedCode := fmt.Sprint(c0["procedure_code"])
	if pinnedVerID != ksVer.String() && pinnedVerID != ctVer.String() {
		t.Fatalf("candidate must pin seed 1.0.0 version, got %s", pinnedVerID)
	}
	wantHash := chat.DefinitionHash(ksPatched)
	if pinnedCode == "chung_thuc_ban_sao" {
		wantHash = chat.DefinitionHash(ctPatched)
	}
	if pinnedHash != wantHash {
		t.Fatalf("candidate hash %s want %s for %s", pinnedHash, wantHash, pinnedCode)
	}

	procID := ksID
	oldVer := ksVer
	oldDef := append([]byte(nil), ksPatched...)
	if pinnedCode == "chung_thuc_ban_sao" {
		procID = ctID
		oldVer = ctVer
		oldDef = append([]byte(nil), ctPatched...)
	}
	var mutated map[string]any
	_ = json.Unmarshal(oldDef, &mutated)
	mutated["guidance"] = map[string]any{
		"summary":         "MUTATED_V2_SHOULD_NOT_APPEAR",
		"checklist":       []string{},
		"where_to_submit": "x",
	}
	mutatedBytes, _ := json.Marshal(mutated)
	v2Label := fmt.Sprintf("1.0.0-v2-confirm-%s", uuid.NewString()[:8])
	var v2 uuid.UUID
	if err := pool.QueryRow(ctx, `
		INSERT INTO procedure_versions (procedure_id, version, status, definition, created_by)
		SELECT procedure_id, $2, 'APPROVED', $3::jsonb, created_by
		FROM procedure_versions WHERE id=$1 RETURNING id`, oldVer, v2Label, mutatedBytes).Scan(&v2); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `UPDATE procedure_versions SET status='ARCHIVED' WHERE id=$1`, oldVer); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `UPDATE procedure_versions SET status='ACTIVE' WHERE id=$1`, v2); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `UPDATE procedures SET active_version_id=$2 WHERE id=$1`, procID, v2); err != nil {
		t.Fatal(err)
	}

	code, yesEnv := apiJSON(t, eng, http.MethodPost, "/api/v1/sessions/"+sessionID+"/turns",
		map[string]any{"message": "Có"},
		map[string]string{"X-Guest-Token": guest, "X-Request-ID": uuid.NewString()})
	if code != http.StatusOK {
		t.Fatalf("yes turn %d %v", code, yesEnv)
	}
	yd := dataMap(yesEnv)
	reply := fmt.Sprint(yd["reply_text"])
	if bytes.Contains([]byte(reply), []byte("MUTATED_V2_SHOULD_NOT_APPEAR")) {
		t.Fatalf("used ACTIVE v2 after confirm pin: %s", reply)
	}
	if fmt.Sprint(yd["procedure_code"]) != pinnedCode {
		t.Fatalf("procedure_code %v want %s (action=%v)", yd["procedure_code"], pinnedCode, yd["action"])
	}

	var sessionVer uuid.UUID
	if err := pool.QueryRow(ctx, `SELECT active_procedure_version_id FROM conversation_sessions WHERE id=$1`, sessionID).Scan(&sessionVer); err != nil {
		t.Fatal(err)
	}
	if sessionVer.String() != pinnedVerID {
		t.Fatalf("session pin %s want candidate v1 %s (ACTIVE is now %s)", sessionVer, pinnedVerID, v2)
	}
	var storedDef []byte
	if err := pool.QueryRow(ctx, `SELECT definition FROM procedure_versions WHERE id=$1`, sessionVer).Scan(&storedDef); err != nil {
		t.Fatal(err)
	}
	if chat.DefinitionHash(storedDef) != pinnedHash {
		t.Fatalf("definition hash mismatch after pin")
	}
	gotMut, _ := pool.Exec(ctx, `SELECT 1 FROM procedure_versions WHERE id=$1`, v2)
	_ = gotMut
	if chat.DefinitionHash(storedDef) == chat.DefinitionHash(mutatedBytes) {
		t.Fatal("session somehow pinned mutated v2")
	}
}

func TestHTTPCitizenVsAdminCatalog(t *testing.T) {
	pool := requireIntegrationDB(t)
	defer pool.Close()
	eng := testEngine(t, pool)
	ctx := context.Background()

	// Public / guest: citizen=true → CitizenDomainIDs only
	code, env := apiJSON(t, eng, http.MethodGet, "/api/v1/procedures?xa_id=xa_chu_se&citizen=true", nil, nil)
	if code != http.StatusOK {
		t.Fatalf("%d %v", code, env)
	}
	citizenDomains, _ := dataMap(env)["domains"].([]any)
	citizenCount := 0
	for _, raw := range citizenDomains {
		g, _ := raw.(map[string]any)
		if fmt.Sprint(g["domain_id"]) != "ho_tich_chung_thuc" {
			t.Fatalf("citizen catalog leaked domain %v", g["domain_id"])
		}
		citizenCount += int(g["count"].(float64))
	}

	// Guest cannot use citizen=false as a privilege flag
	code, env = apiJSON(t, eng, http.MethodGet, "/api/v1/procedures?xa_id=xa_chu_se&citizen=false", nil, nil)
	if code != http.StatusUnauthorized {
		t.Fatalf("guest full catalog want 401 got %d %v", code, env)
	}

	// Invalid boolean → 400 (not admin mode)
	code, env = apiJSON(t, eng, http.MethodGet, "/api/v1/procedures?xa_id=xa_chu_se&citizen=maybe", nil, nil)
	if code != http.StatusBadRequest {
		t.Fatalf("invalid citizen want 400 got %d %v", code, env)
	}

	hashC, err := auth.HashPassword("citizen-pass-long")
	if err != nil {
		t.Fatal(err)
	}
	hashA, err := auth.HashPassword("admin-pass-longgg")
	if err != nil {
		t.Fatal(err)
	}
	citizenID := uuid.New()
	adminID := uuid.New()
	_, err = pool.Exec(ctx, `
		INSERT INTO users (id, role, full_name, email, password_hash)
		VALUES
		 ($1,'CITIZEN','Cat Citizen',$2,$3),
		 ($4,'ADMIN','Cat Admin',$5,$6)`,
		citizenID, "cat-citizen-"+citizenID.String()[:8]+"@example.com", hashC,
		adminID, "cat-admin-"+adminID.String()[:8]+"@example.com", hashA)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_, _ = pool.Exec(ctx, `DELETE FROM users WHERE id IN ($1,$2)`, citizenID, adminID)
	})

	tokSvc := &auth.TokenService{Secret: []byte("integration-test-jwt-secret-key"), TTL: time.Hour}
	citizenTok, _, err := tokSvc.Issue(citizenID, auth.RoleCitizen, "c@example.com", "C")
	if err != nil {
		t.Fatal(err)
	}
	adminTok, _, err := tokSvc.Issue(adminID, auth.RoleAdmin, "a@example.com", "A")
	if err != nil {
		t.Fatal(err)
	}

	// Citizen JWT + citizen=false → 403
	code, env = apiJSON(t, eng, http.MethodGet, "/api/v1/procedures?xa_id=xa_chu_se&citizen=false", nil,
		map[string]string{"Authorization": "Bearer " + citizenTok})
	if code != http.StatusForbidden {
		t.Fatalf("citizen full catalog want 403 got %d %v", code, env)
	}

	// Admin JWT + citizen=false → full catalog
	code, env = apiJSON(t, eng, http.MethodGet, "/api/v1/procedures?xa_id=xa_chu_se&citizen=false", nil,
		map[string]string{"Authorization": "Bearer " + adminTok})
	if code != http.StatusOK {
		t.Fatalf("admin catalog %d %v", code, env)
	}
	adminDomains, _ := dataMap(env)["domains"].([]any)
	adminCount := int(dataMap(env)["count"].(float64))
	if adminCount <= citizenCount {
		t.Fatalf("admin catalog should include more than citizen: admin=%d citizen=%d domains=%d", adminCount, citizenCount, len(adminDomains))
	}
	seenExtra := false
	for _, raw := range adminDomains {
		g, _ := raw.(map[string]any)
		if fmt.Sprint(g["domain_id"]) != "ho_tich_chung_thuc" {
			seenExtra = true
		}
	}
	if !seenExtra {
		t.Fatal("admin catalog missing non-citizen domains")
	}
}

func TestHTTPProcedureDetailScopeAuth(t *testing.T) {
	pool := requireIntegrationDB(t)
	defer pool.Close()
	eng := testEngine(t, pool)
	ctx := context.Background()

	// Citizen-scope procedure (domain ho_tich_chung_thuc)
	inCode := "dk_khai_sinh"
	// Outside citizen scope (domain bao_hiem_chinh_sach_xh)
	outCode := "dk_bhyt_ho_gia_dinh"

	var inID, outID uuid.UUID
	if err := pool.QueryRow(ctx, `SELECT id FROM procedures WHERE xa_id='xa_chu_se' AND procedure_code=$1`, inCode).Scan(&inID); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(ctx, `SELECT id FROM procedures WHERE xa_id='xa_chu_se' AND procedure_code=$1`, outCode).Scan(&outID); err != nil {
		t.Fatal(err)
	}

	hashC, err := auth.HashPassword("citizen-detail-pass")
	if err != nil {
		t.Fatal(err)
	}
	hashA, err := auth.HashPassword("admin-detail-passs")
	if err != nil {
		t.Fatal(err)
	}
	citizenID := uuid.New()
	adminID := uuid.New()
	_, err = pool.Exec(ctx, `
		INSERT INTO users (id, role, full_name, email, password_hash)
		VALUES
		 ($1,'CITIZEN','Detail Citizen',$2,$3),
		 ($4,'ADMIN','Detail Admin',$5,$6)`,
		citizenID, "det-citizen-"+citizenID.String()[:8]+"@example.com", hashC,
		adminID, "det-admin-"+adminID.String()[:8]+"@example.com", hashA)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_, _ = pool.Exec(ctx, `DELETE FROM users WHERE id IN ($1,$2)`, citizenID, adminID)
	})
	tokSvc := &auth.TokenService{Secret: []byte("integration-test-jwt-secret-key"), TTL: time.Hour}
	citizenTok, _, err := tokSvc.Issue(citizenID, auth.RoleCitizen, "dc@example.com", "C")
	if err != nil {
		t.Fatal(err)
	}
	adminTok, _, err := tokSvc.Issue(adminID, auth.RoleAdmin, "da@example.com", "A")
	if err != nil {
		t.Fatal(err)
	}
	adminHdr := map[string]string{"Authorization": "Bearer " + adminTok}
	citizenHdr := map[string]string{"Authorization": "Bearer " + citizenTok}

	byCode := func(code string) string {
		return "/api/v1/procedures/by-code/" + code + "?xa_id=xa_chu_se"
	}
	active := func(id uuid.UUID) string {
		return "/api/v1/procedures/" + id.String() + "/active-version"
	}

	// Guest: in-scope OK
	code, env := apiJSON(t, eng, http.MethodGet, byCode(inCode), nil, nil)
	if code != http.StatusOK {
		t.Fatalf("guest in-scope by-code want 200 got %d %v", code, env)
	}
	code, env = apiJSON(t, eng, http.MethodGet, active(inID), nil, nil)
	if code != http.StatusOK {
		t.Fatalf("guest in-scope active-version want 200 got %d %v", code, env)
	}

	// Guest: out-of-scope denied (404 — no existence leak)
	code, env = apiJSON(t, eng, http.MethodGet, byCode(outCode), nil, nil)
	if code != http.StatusNotFound {
		t.Fatalf("guest out-of-scope by-code want 404 got %d %v", code, env)
	}
	code, env = apiJSON(t, eng, http.MethodGet, active(outID), nil, nil)
	if code != http.StatusNotFound {
		t.Fatalf("guest out-of-scope active-version want 404 got %d %v", code, env)
	}

	// Citizen JWT: in-scope OK, out-of-scope 403
	code, env = apiJSON(t, eng, http.MethodGet, byCode(inCode), nil, citizenHdr)
	if code != http.StatusOK {
		t.Fatalf("citizen in-scope by-code want 200 got %d %v", code, env)
	}
	code, env = apiJSON(t, eng, http.MethodGet, active(inID), nil, citizenHdr)
	if code != http.StatusOK {
		t.Fatalf("citizen in-scope active-version want 200 got %d %v", code, env)
	}
	code, env = apiJSON(t, eng, http.MethodGet, byCode(outCode), nil, citizenHdr)
	if code != http.StatusForbidden {
		t.Fatalf("citizen out-of-scope by-code want 403 got %d %v", code, env)
	}
	code, env = apiJSON(t, eng, http.MethodGet, active(outID), nil, citizenHdr)
	if code != http.StatusForbidden {
		t.Fatalf("citizen out-of-scope active-version want 403 got %d %v", code, env)
	}

	// Admin JWT: out-of-scope OK (full catalog detail)
	code, env = apiJSON(t, eng, http.MethodGet, byCode(outCode), nil, adminHdr)
	if code != http.StatusOK {
		t.Fatalf("admin out-of-scope by-code want 200 got %d %v", code, env)
	}
	if fmt.Sprint(dataMap(env)["procedure_code"]) != outCode {
		t.Fatalf("admin by-code data=%v", env)
	}
	code, env = apiJSON(t, eng, http.MethodGet, active(outID), nil, adminHdr)
	if code != http.StatusOK {
		t.Fatalf("admin out-of-scope active-version want 200 got %d %v", code, env)
	}
	if dataMap(env)["definition"] == nil {
		t.Fatalf("admin active-version missing definition: %v", env)
	}
}

func TestHTTPHistoryLatestN(t *testing.T) {
	pool := requireIntegrationDB(t)
	defer pool.Close()
	eng := testEngine(t, pool)

	_, env := apiJSON(t, eng, http.MethodPost, "/api/v1/sessions", map[string]any{"xa_id": "xa_chu_se"}, nil)
	d := dataMap(env)
	sessionID, _ := d["id"].(string)
	guest, _ := d["guest_token"].(string)

	var lastIDs []string
	for i := 0; i < 3; i++ {
		_, turnEnv := apiJSON(t, eng, http.MethodPost, "/api/v1/sessions/"+sessionID+"/turns",
			map[string]any{"message": fmt.Sprintf("Đăng ký khai sinh lần %d", i)},
			map[string]string{"X-Guest-Token": guest, "X-Request-ID": uuid.NewString()})
		td := dataMap(turnEnv)
		um, _ := td["user_message"].(map[string]any)
		am, _ := td["assistant_message"].(map[string]any)
		lastIDs = append(lastIDs, fmt.Sprint(um["id"]), fmt.Sprint(am["id"]))
	}

	code, histEnv := apiJSON(t, eng, http.MethodGet, "/api/v1/sessions/"+sessionID+"/messages?limit=2", nil,
		map[string]string{"X-Guest-Token": guest})
	if code != http.StatusOK {
		t.Fatalf("history %d", code)
	}
	items, _ := dataMap(histEnv)["items"].([]any)
	if len(items) != 2 {
		t.Fatalf("want 2 got %d", len(items))
	}
	wantTail := lastIDs[len(lastIDs)-2:]
	got0, _ := items[0].(map[string]any)
	got1, _ := items[1].(map[string]any)
	gotIDs := []string{fmt.Sprint(got0["id"]), fmt.Sprint(got1["id"])}
	if gotIDs[0] != wantTail[0] || gotIDs[1] != wantTail[1] {
		t.Fatalf("latest-N mismatch want %v got %v", wantTail, gotIDs)
	}
}
