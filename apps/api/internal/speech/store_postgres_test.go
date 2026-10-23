package speech

import (
	"context"
	"os"
	"path/filepath"
	"strings"
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

	execStatements(t, ctx, pool, readMigration(t, "000021_speech_translation_requests.up.sql"))

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
	execStatements(t, ctx, pool, readMigration(t, "000021_speech_translation_requests.up.sql"))
	if err := (&Store{Pool: pool}).Insert(ctx, Record{
		RequestID: uuid.New(), SessionID: sess, XaID: "xa_fresh",
		Outcome: OutcomeSuccess, HTTPStatus: 200, ModelID: "c0_cascaded",
	}); err != nil {
		t.Fatal(err)
	}
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

func isolatedSpeechSchema(t *testing.T) (context.Context, *pgxpool.Pool, string) {
	t.Helper()
	dsn := os.Getenv("DATABASE_URL")
	if dsn == "" {
		dsn = "postgres://cas:cas@127.0.0.1:5432/citizen_assistance?sslmode=disable"
	}
	schema := "h3_" + strings.ReplaceAll(uuid.NewString(), "-", "")[:12]
	cfg, err := pgxpool.ParseConfig(dsn)
	if err != nil {
		t.Skipf("postgres dsn: %v", err)
	}
	cfg.AfterConnect = func(ctx context.Context, conn *pgx.Conn) error {
		_, err := conn.Exec(ctx, "SET search_path TO "+schema)
		return err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 8*time.Second)
	t.Cleanup(cancel)
	admin, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Skipf("postgres unavailable: %v", err)
	}
	if err := admin.Ping(ctx); err != nil {
		admin.Close()
		t.Skipf("postgres unreachable: %v", err)
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
