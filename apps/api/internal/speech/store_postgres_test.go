package speech

import (
	"context"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestStoreInsertPostgres(t *testing.T) {
	ctx, pool, schema := isolatedSpeechSchema(t)
	execStatements(t, ctx, pool, legacySpeechDDL())
	xaA, xaB := "xa_h3_a", "xa_h3_b"
	mustExec(t, ctx, pool, `INSERT INTO communes (id, name) VALUES ($1,'A'), ($2,'B')`, xaA, xaB)
	legacyID := uuid.New()
	sessionLegacy := uuid.New()
	mustExec(t, ctx, pool, `
		INSERT INTO conversation_sessions (id, xa_id, status) VALUES ($1,$2,'OPEN')`, sessionLegacy, xaA)
	mustExec(t, ctx, pool, `
		INSERT INTO speech_translation_requests (
			id, request_id, session_id, transcript_bahnar, output_vi, status
		) VALUES ($1,$2,$3,'legacy-bahnar','legacy-vi','OK')`,
		legacyID, uuid.New(), sessionLegacy)

	applySpeechAuditMigrations(t, ctx, pool)

	var transcript, output, xaBackfill string
	if err := pool.QueryRow(ctx, `
		SELECT transcript_bahnar, output_vi, xa_id
		FROM speech_translation_requests WHERE id = $1`, legacyID).
		Scan(&transcript, &output, &xaBackfill); err != nil {
		t.Fatal(err)
	}
	if transcript != "legacy-bahnar" || output != "legacy-vi" {
		t.Fatalf("legacy transcripts destroyed: %q %q", transcript, output)
	}
	if xaBackfill != xaA {
		t.Fatalf("legacy xa_id backfill %q", xaBackfill)
	}

	sessionNew := uuid.New()
	mustExec(t, ctx, pool, `
		INSERT INTO conversation_sessions (id, xa_id, status) VALUES ($1,$2,'OPEN')`, sessionNew, xaB)
	store := &Store{Pool: pool}
	rid := uuid.New()
	flag := true
	dur, lat := 14432, 529
	if err := store.Insert(ctx, Record{
		RequestID: rid, SessionID: sessionNew, XaID: xaB,
		Outcome: OutcomeSuccess, HTTPStatus: 200, UpstreamStatus: 200,
		DurationMS: &dur, LatencyMS: &lat, ModelID: "c0_cascaded",
		TokenizerParity: "VERIFIED", ResearchProvenance: "NOT_VERIFIED",
		CheckpointIdentity: "expected_step_match", GenerationMatchesRQ1: &flag,
		ASRCheckpointStep: intPtr(12500), MTCheckpointStep: intPtr(17500),
	}); err != nil {
		t.Fatal(err)
	}

	var outcome, model, provenance, status, gotXa, tokParity string
	var httpStatus, asrStep, mtStep int
	var audio, asr, vi *string
	if err := pool.QueryRow(ctx, `
		SELECT outcome, model_id, research_provenance, http_status, status,
		       audio_uri, transcript_bahnar, output_vi, xa_id, tokenizer_parity,
		       asr_checkpoint_step, mt_checkpoint_step
		FROM speech_translation_requests WHERE request_id = $1`, rid).
		Scan(&outcome, &model, &provenance, &httpStatus, &status, &audio, &asr, &vi, &gotXa, &tokParity, &asrStep, &mtStep); err != nil {
		t.Fatal(err)
	}
	if outcome != OutcomeSuccess || model != "c0_cascaded" || provenance != "NOT_VERIFIED" || httpStatus != 200 || status != "OK" {
		t.Fatalf("new row %s %s %s %d %s", outcome, model, provenance, httpStatus, status)
	}
	if gotXa != xaB || tokParity != "VERIFIED" || asrStep != 12500 || mtStep != 17500 {
		t.Fatalf("metadata xa=%s parity=%s asr=%d mt=%d", gotXa, tokParity, asrStep, mtStep)
	}
	if audio != nil || asr != nil || vi != nil {
		t.Fatalf("new row stored audio/transcript: %v %v %v", audio, asr, vi)
	}

	failID := uuid.New()
	if err := store.Insert(ctx, Record{
		RequestID: failID, SessionID: sessionNew, XaID: xaB,
		Outcome: OutcomeFailure, ErrorCode: "SPEECH_UNAVAILABLE", HTTPStatus: 502,
	}); err != nil {
		t.Fatal(err)
	}
	var failOutcome, failCode, failStatus string
	if err := pool.QueryRow(ctx, `
		SELECT outcome, error_code, status FROM speech_translation_requests WHERE request_id = $1`, failID).
		Scan(&failOutcome, &failCode, &failStatus); err != nil {
		t.Fatal(err)
	}
	if failOutcome != OutcomeFailure || failCode != "SPEECH_UNAVAILABLE" || failStatus != "FAILED" {
		t.Fatalf("failure row %s %s %s", failOutcome, failCode, failStatus)
	}

	bad := store.Insert(ctx, Record{
		RequestID: uuid.New(), SessionID: uuid.New(), XaID: "no-such-commune",
		Outcome: OutcomeSuccess, HTTPStatus: 200,
	})
	if bad == nil {
		t.Fatal("expected commune FK violation")
	}

	execStatements(t, ctx, pool, readMigration(t, "000022_speech_chat_turn_unique.down.sql"))
	execStatements(t, ctx, pool, readMigration(t, "000021_speech_translation_requests.down.sql"))
	var hasXa bool
	if err := pool.QueryRow(ctx, `
		SELECT EXISTS (
			SELECT 1 FROM information_schema.columns
			WHERE table_schema = $1 AND table_name = 'speech_translation_requests' AND column_name = 'xa_id'
		)`, schema).Scan(&hasXa); err != nil {
		t.Fatal(err)
	}
	if hasXa {
		t.Fatal("down migration left xa_id")
	}
	var still string
	if err := pool.QueryRow(ctx, `SELECT transcript_bahnar FROM speech_translation_requests WHERE id = $1`, legacyID).Scan(&still); err != nil {
		t.Fatalf("down dropped 000005 row: %v", err)
	}
	if still != "legacy-bahnar" {
		t.Fatalf("down changed legacy transcript %q", still)
	}
	_ = ctx
}

func TestStoreInsertPostgresFreshAfter000005(t *testing.T) {
	ctx, pool, _ := isolatedSpeechSchema(t)
	execStatements(t, ctx, pool, legacySpeechDDL())
	mustExec(t, ctx, pool, `INSERT INTO communes (id, name) VALUES ('xa_fresh','Fresh')`)
	sess := uuid.New()
	mustExec(t, ctx, pool, `INSERT INTO conversation_sessions (id, xa_id, status) VALUES ($1,'xa_fresh','OPEN')`, sess)
	applySpeechAuditMigrations(t, ctx, pool)
	if err := (&Store{Pool: pool}).Insert(ctx, Record{
		RequestID: uuid.New(), SessionID: sess, XaID: "xa_fresh",
		Outcome: OutcomeSuccess, HTTPStatus: 200, ModelID: "c0_cascaded",
	}); err != nil {
		t.Fatal(err)
	}
}

func TestStoreLinkChatTurnRequiresExplicitIDs(t *testing.T) {
	ctx, pool, _ := isolatedSpeechSchema(t)
	execStatements(t, ctx, pool, legacySpeechDDL())
	mustExec(t, ctx, pool, `INSERT INTO communes (id, name) VALUES ('xa_link','Link')`)
	sessA, sessB := uuid.New(), uuid.New()
	mustExec(t, ctx, pool, `INSERT INTO conversation_sessions (id, xa_id, status) VALUES ($1,'xa_link','OPEN'), ($2,'xa_link','OPEN')`, sessA, sessB)
	applySpeechAuditMigrations(t, ctx, pool)
	store := &Store{Pool: pool}
	successID, failID, otherSessID := uuid.New(), uuid.New(), uuid.New()
	if err := store.Insert(ctx, Record{
		RequestID: successID, SessionID: sessA, XaID: "xa_link",
		Outcome: OutcomeSuccess, HTTPStatus: 200, ModelID: "c0_cascaded",
	}); err != nil {
		t.Fatal(err)
	}
	if err := store.Insert(ctx, Record{
		RequestID: failID, SessionID: sessA, XaID: "xa_link",
		Outcome: OutcomeFailure, ErrorCode: "SPEECH_NO_SPEECH", HTTPStatus: 422,
	}); err != nil {
		t.Fatal(err)
	}
	if err := store.Insert(ctx, Record{
		RequestID: otherSessID, SessionID: sessB, XaID: "xa_link",
		Outcome: OutcomeSuccess, HTTPStatus: 200, ModelID: "c0_cascaded",
	}); err != nil {
		t.Fatal(err)
	}

	turnID := uuid.New()
	linked, err := store.LinkChatTurn(ctx, successID, sessA, turnID)
	if err != nil || linked.Outcome != LinkLinked {
		t.Fatalf("link %s %v", linked.Outcome, err)
	}
	wrongTurn := uuid.New()
	wrongSess, err := store.LinkChatTurn(ctx, successID, sessB, wrongTurn)
	if err != nil || wrongSess.Outcome != LinkConflict {
		t.Fatalf("other session %s %v", wrongSess.Outcome, err)
	}
	failOut, err := store.LinkChatTurn(ctx, failID, sessA, uuid.New())
	if err != nil || failOut.Outcome != LinkConflict {
		t.Fatalf("failure row %s %v", failOut.Outcome, err)
	}
	otherOut, err := store.LinkChatTurn(ctx, otherSessID, sessA, uuid.New())
	if err != nil || otherOut.Outcome != LinkConflict {
		t.Fatalf("other-session row %s %v", otherOut.Outcome, err)
	}
	missing, err := store.LinkChatTurn(ctx, uuid.New(), sessA, uuid.New())
	if err != nil || missing.Outcome != LinkNotFound {
		t.Fatalf("missing %s %v", missing.Outcome, err)
	}

	var stored *uuid.UUID
	if err := pool.QueryRow(ctx, `
		SELECT chat_turn_request_id FROM speech_translation_requests WHERE request_id = $1`, successID).
		Scan(&stored); err != nil {
		t.Fatal(err)
	}
	if stored == nil || *stored != turnID {
		t.Fatalf("success row link %v want %s", stored, turnID)
	}

	var failLink *uuid.UUID
	if err := pool.QueryRow(ctx, `
		SELECT chat_turn_request_id FROM speech_translation_requests WHERE request_id = $1`, failID).
		Scan(&failLink); err != nil {
		t.Fatal(err)
	}
	if failLink != nil {
		t.Fatalf("failure row was linked %v", failLink)
	}

	var otherLink *uuid.UUID
	if err := pool.QueryRow(ctx, `
		SELECT chat_turn_request_id FROM speech_translation_requests WHERE request_id = $1`, otherSessID).
		Scan(&otherLink); err != nil {
		t.Fatal(err)
	}
	if otherLink != nil {
		t.Fatalf("other-session row was linked by timestamp-style mismatch: %v", otherLink)
	}

	retry, err := store.LinkChatTurn(ctx, successID, sessA, turnID)
	if err != nil || retry.Outcome != LinkAlreadyLinked {
		t.Fatalf("idempotent retry %s %v", retry.Outcome, err)
	}
	second := uuid.New()
	overwrite, err := store.LinkChatTurn(ctx, successID, sessA, second)
	if err != nil || overwrite.Outcome != LinkConflict {
		t.Fatalf("overwrite %s %v", overwrite.Outcome, err)
	}
	reuseTurn, err := store.LinkChatTurn(ctx, otherSessID, sessB, turnID)
	if err != nil || reuseTurn.Outcome != LinkConflict {
		t.Fatalf("turn reused on other speech %s %v", reuseTurn.Outcome, err)
	}
	var still *uuid.UUID
	if err := pool.QueryRow(ctx, `
		SELECT chat_turn_request_id FROM speech_translation_requests WHERE request_id = $1`, successID).
		Scan(&still); err != nil {
		t.Fatal(err)
	}
	if still == nil || *still != turnID {
		t.Fatalf("existing link overwritten %v", still)
	}

	var asr, vi *string
	if err := pool.QueryRow(ctx, `
		SELECT transcript_bahnar, output_vi FROM speech_translation_requests WHERE request_id = $1`, successID).
		Scan(&asr, &vi); err != nil {
		t.Fatal(err)
	}
	if asr != nil || vi != nil {
		t.Fatalf("link wrote transcripts %v %v", asr, vi)
	}
	t.Log("postgres_link_chat_turn_executed=true")
}

func TestStoreLinkChatTurnConcurrentDifferentTurns(t *testing.T) {
	ctx, pool, _ := isolatedSpeechSchema(t)
	execStatements(t, ctx, pool, legacySpeechDDL())
	mustExec(t, ctx, pool, `INSERT INTO communes (id, name) VALUES ('xa_race','Race')`)
	sess := uuid.New()
	mustExec(t, ctx, pool, `INSERT INTO conversation_sessions (id, xa_id, status) VALUES ($1,'xa_race','OPEN')`, sess)
	applySpeechAuditMigrations(t, ctx, pool)
	store := &Store{Pool: pool}
	speechID := uuid.New()
	if err := store.Insert(ctx, Record{
		RequestID: speechID, SessionID: sess, XaID: "xa_race",
		Outcome: OutcomeSuccess, HTTPStatus: 200, ModelID: "c0_cascaded",
	}); err != nil {
		t.Fatal(err)
	}
	turnA, turnB := uuid.New(), uuid.New()
	var wg sync.WaitGroup
	outcomes := make([]LinkOutcome, 2)
	errs := make([]error, 2)
	ready := make(chan struct{})
	wg.Add(2)
	go func() {
		defer wg.Done()
		<-ready
		out, err := store.LinkChatTurn(ctx, speechID, sess, turnA)
		outcomes[0], errs[0] = out.Outcome, err
	}()
	go func() {
		defer wg.Done()
		<-ready
		out, err := store.LinkChatTurn(ctx, speechID, sess, turnB)
		outcomes[1], errs[1] = out.Outcome, err
	}()
	close(ready)
	wg.Wait()
	if errs[0] != nil || errs[1] != nil {
		t.Fatalf("errs %v %v", errs[0], errs[1])
	}
	var linked, conflict int
	for _, o := range outcomes {
		switch o {
		case LinkLinked:
			linked++
		case LinkConflict, LinkAlreadyLinked:
			conflict++
		default:
			t.Fatalf("unexpected outcome %s", o)
		}
	}
	if linked != 1 || conflict != 1 {
		t.Fatalf("outcomes %v", outcomes)
	}
	t.Log("postgres_link_chat_turn_concurrent_executed=true")
}

func TestStoreLinkChatTurnConcurrentTwoSpeechOneTurn(t *testing.T) {
	ctx, pool, _ := isolatedSpeechSchema(t)
	execStatements(t, ctx, pool, legacySpeechDDL())
	mustExec(t, ctx, pool, `INSERT INTO communes (id, name) VALUES ('xa_one','One')`)
	sess := uuid.New()
	mustExec(t, ctx, pool, `INSERT INTO conversation_sessions (id, xa_id, status) VALUES ($1,'xa_one','OPEN')`, sess)
	applySpeechAuditMigrations(t, ctx, pool)
	store := &Store{Pool: pool}
	speechA, speechB := uuid.New(), uuid.New()
	for _, id := range []uuid.UUID{speechA, speechB} {
		if err := store.Insert(ctx, Record{
			RequestID: id, SessionID: sess, XaID: "xa_one",
			Outcome: OutcomeSuccess, HTTPStatus: 200, ModelID: "c0_cascaded",
		}); err != nil {
			t.Fatal(err)
		}
	}
	turnID := uuid.New()
	var wg sync.WaitGroup
	outcomes := make([]LinkOutcome, 2)
	errs := make([]error, 2)
	ready := make(chan struct{})
	wg.Add(2)
	go func() {
		defer wg.Done()
		<-ready
		out, err := store.LinkChatTurn(ctx, speechA, sess, turnID)
		outcomes[0], errs[0] = out.Outcome, err
	}()
	go func() {
		defer wg.Done()
		<-ready
		out, err := store.LinkChatTurn(ctx, speechB, sess, turnID)
		outcomes[1], errs[1] = out.Outcome, err
	}()
	close(ready)
	wg.Wait()
	if errs[0] != nil || errs[1] != nil {
		t.Fatalf("errs %v %v", errs[0], errs[1])
	}
	var linked, conflict int
	for _, o := range outcomes {
		switch o {
		case LinkLinked:
			linked++
		case LinkConflict, LinkAlreadyLinked:
			conflict++
		default:
			t.Fatalf("unexpected outcome %s", o)
		}
	}
	if linked != 1 || conflict != 1 {
		t.Fatalf("two-speech one-turn outcomes %v", outcomes)
	}
	var n int
	if err := pool.QueryRow(ctx, `
		SELECT COUNT(*) FROM speech_translation_requests WHERE chat_turn_request_id = $1`, turnID).
		Scan(&n); err != nil {
		t.Fatal(err)
	}
	if n != 1 {
		t.Fatalf("unique chat_turn_request_id violated: %d rows", n)
	}
	t.Log("postgres_link_two_speech_one_turn_executed=true")
}

func TestStoreInsertPostgresMissingMigrationFails(t *testing.T) {
	ctx, pool, _ := isolatedSpeechSchema(t)
	execStatements(t, ctx, pool, legacySpeechDDL())
	mustExec(t, ctx, pool, `INSERT INTO communes (id, name) VALUES ('xa_miss','M')`)
	sess := uuid.New()
	mustExec(t, ctx, pool, `INSERT INTO conversation_sessions (id, xa_id, status) VALUES ($1,'xa_miss','OPEN')`, sess)
	err := (&Store{Pool: pool}).Insert(ctx, Record{
		RequestID: uuid.New(), SessionID: sess, XaID: "xa_miss",
		Outcome: OutcomeSuccess, HTTPStatus: 200,
	})
	if err == nil {
		t.Fatal("insert against 000005-only schema must fail")
	}
}

func dialSpeechTestPool(ctx context.Context, dsn string) (*pgxpool.Pool, error) {
	cfg, err := pgxpool.ParseConfig(dsn)
	if err != nil {
		return nil, err
	}
	pool, err := pgxpool.NewWithConfig(ctx, cfg)
	if err != nil {
		return nil, err
	}
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, err
	}
	return pool, nil
}

func isolatedSpeechSchema(t *testing.T) (context.Context, *pgxpool.Pool, string) {
	t.Helper()
	dsn := speechTestDSN(t)
	schema := "h3_" + strings.ReplaceAll(uuid.NewString(), "-", "")[:12]
	cfg, err := pgxpool.ParseConfig(dsn)
	if err != nil {
		t.Fatalf("SPEECH_TEST_DATABASE_URL is set but DSN is unusable: %v", err)
	}
	cfg.AfterConnect = func(ctx context.Context, conn *pgx.Conn) error {
		_, err := conn.Exec(ctx, "SET search_path TO "+schema)
		return err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 8*time.Second)
	t.Cleanup(cancel)
	admin, err := dialSpeechTestPool(ctx, dsn)
	if err != nil {
		t.Fatalf("SPEECH_TEST_DATABASE_URL is set but postgres is unreachable: %v", err)
	}
	_, _ = admin.Exec(ctx, "CREATE EXTENSION IF NOT EXISTS pgcrypto")
	if _, err := admin.Exec(ctx, "CREATE SCHEMA "+schema); err != nil {
		admin.Close()
		t.Fatalf("create schema: %v", err)
	}
	t.Cleanup(func() {
		clean, ccancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer ccancel()
		_, _ = admin.Exec(clean, "DROP SCHEMA IF EXISTS "+schema+" CASCADE")
		admin.Close()
	})
	pool, err := pgxpool.NewWithConfig(ctx, cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(pool.Close)
	return ctx, pool, schema
}

func legacySpeechDDL() string {
	return `
CREATE TABLE communes (
    id text PRIMARY KEY,
    name text NOT NULL
);
CREATE TABLE users (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid()
);
CREATE TABLE conversation_sessions (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    xa_id text NOT NULL REFERENCES communes (id),
    status text NOT NULL DEFAULT 'OPEN'
);
CREATE TABLE model_versions (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid()
);
CREATE TABLE speech_translation_requests (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id uuid NOT NULL,
    session_id uuid REFERENCES conversation_sessions (id) ON DELETE SET NULL,
    model_version_id uuid REFERENCES model_versions (id) ON DELETE RESTRICT,
    audio_uri text,
    transcript_bahnar text,
    output_vi text,
    latency_ms int,
    status text NOT NULL DEFAULT 'QUEUED',
    created_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT speech_translation_requests_status_check CHECK (
        status IN ('QUEUED', 'OK', 'FAILED')
    )
);
`
}

func applySpeechAuditMigrations(t *testing.T, ctx context.Context, pool *pgxpool.Pool) {
	t.Helper()
	execStatements(t, ctx, pool, readMigration(t, "000021_speech_translation_requests.up.sql"))
	execStatements(t, ctx, pool, readMigration(t, "000022_speech_chat_turn_unique.up.sql"))
}

func speechTestDSN(t *testing.T) string {
	t.Helper()
	dsn := strings.TrimSpace(os.Getenv("SPEECH_TEST_DATABASE_URL"))
	if dsn == "" && os.Getenv("CITIZEN_SPEECH_TEST_ALLOW_DSN") == "1" {
		dsn = strings.TrimSpace(os.Getenv("DATABASE_URL"))
	}
	if dsn == "" {
		t.Skip("SPEECH_TEST_DATABASE_URL required (or DATABASE_URL with CITIZEN_SPEECH_TEST_ALLOW_DSN=1); refusing implicit citizen_assistance fallback")
	}
	if err := validateSpeechTestDSN(dsn); err != nil {
		t.Fatal(err)
	}
	return dsn
}

func validateSpeechTestDSN(dsn string) error {
	u, err := url.Parse(dsn)
	if err != nil || u.Scheme == "" || u.Host == "" {
		return fmt.Errorf("invalid speech test DSN")
	}
	host := strings.ToLower(u.Hostname())
	loopback := host == "127.0.0.1" || host == "localhost" || host == "::1"
	if !loopback && os.Getenv("CITIZEN_SPEECH_TEST_ALLOW_REMOTE") != "1" {
		return fmt.Errorf("refusing non-loopback speech test DSN host %q; set CITIZEN_SPEECH_TEST_ALLOW_REMOTE=1 only for an isolated test database", host)
	}
	return nil
}

func readMigration(t *testing.T, name string) string {
	t.Helper()
	p := filepath.Join("..", "..", "..", "..", "deploy", "migrations", name)
	raw, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	return string(raw)
}

func mustExec(t *testing.T, ctx context.Context, pool *pgxpool.Pool, sql string, args ...any) {
	t.Helper()
	if _, err := pool.Exec(ctx, sql, args...); err != nil {
		t.Fatalf("exec: %v\nSQL: %s", err, sql)
	}
}

func execStatements(t *testing.T, ctx context.Context, pool *pgxpool.Pool, sql string) {
	t.Helper()
	var buf strings.Builder
	flush := func() {
		stmt := strings.TrimSpace(buf.String())
		buf.Reset()
		if stmt == "" {
			return
		}
		mustExec(t, ctx, pool, stmt)
	}
	for _, line := range strings.Split(sql, "\n") {
		if strings.HasPrefix(strings.TrimSpace(line), "--") {
			continue
		}
		buf.WriteString(line)
		buf.WriteByte('\n')
		if strings.HasSuffix(strings.TrimSpace(buf.String()), ";") {
			flush()
		}
	}
	flush()
}

func intPtr(v int) *int { return &v }
