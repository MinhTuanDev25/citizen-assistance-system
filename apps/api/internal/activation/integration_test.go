//go:build integration

package activation_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/activation"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/auth"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/config"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/httpserver"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/repository"
	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
)

const seedAdmin = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

func requireP4DB(t *testing.T) *pgxpool.Pool {
	t.Helper()
	url := os.Getenv("CAS_P4_TEST_DATABASE_URL")
	if err := activation.ValidateP4TestDSN(url, os.Getenv); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	pool, err := pgxpool.New(ctx, url)
	if err != nil {
		t.Fatal(err)
	}
	if err := pool.Ping(ctx); err != nil {
		t.Fatal(err)
	}
	var current string
	if err := pool.QueryRow(ctx, `SELECT current_database()`).Scan(&current); err != nil {
		pool.Close()
		t.Fatal(err)
	}
	if current != activation.P4TestDatabase {
		pool.Close()
		t.Fatalf("connected to %s, want %s", current, activation.P4TestDatabase)
	}
	t.Cleanup(pool.Close)
	return pool
}

func testEngine(t *testing.T, pool *pgxpool.Pool) (http.Handler, string, string) {
	t.Helper()
	cfg := config.Config{
		Env: "local", XAID: "xa_chu_se", LogLevel: "error",
		CitizenDomainIDs: []string{"ho_tich_chung_thuc"},
		JWTSecret:        "integration-test-jwt-secret-key", JWTExpireHours: 24,
		DBMaxConns: 8, DBMinConns: 1,
	}
	tokens := &auth.TokenService{Secret: []byte(cfg.JWTSecret), TTL: time.Hour}
	adminTok, _, err := tokens.Issue(uuid.MustParse(seedAdmin), auth.RoleAdmin, "admin@chuse.vn", "Seed Admin")
	if err != nil {
		t.Fatal(err)
	}
	citizenTok, _, err := tokens.Issue(uuid.New(), auth.RoleCitizen, "c@example.com", "Citizen")
	if err != nil {
		t.Fatal(err)
	}
	eng := httpserver.New(slog.New(slog.NewTextHandler(io.Discard, nil)), pool, cfg)
	return eng, adminTok, citizenTok
}

type fixture struct {
	ProcedureID uuid.UUID
	V1          uuid.UUID
	V2          uuid.UUID
	DocID       uuid.UUID
	svc         *activation.Service
	pool        *pgxpool.Pool
}

func validDef(code, name, version string) []byte {
	return []byte(fmt.Sprintf(`{
		"procedure_code": %q,
		"name": %q,
		"version": %q,
		"xa_id": "xa_chu_se",
		"domain": "ho_tich_chung_thuc",
		"status": "APPROVED",
		"intent_examples": ["test p4"],
		"slots": {},
		"required_slots": [],
		"guidance": {"summary": "s", "checklist": ["a"], "where_to_submit": "UBND"}
	}`, code, name, version))
}

func sha(id uuid.UUID) string {
	raw := strings.ReplaceAll(id.String(), "-", "")
	return raw + raw
}

func insertReadySource(t *testing.T, pool *pgxpool.Pool, procID, versionID, docID uuid.UUID) {
	t.Helper()
	ctx := context.Background()
	admin := uuid.MustParse(seedAdmin)
	jobID := uuid.New()
	genID := uuid.New()
	if _, err := pool.Exec(ctx, `
		INSERT INTO documents (
			id, xa_id, domain_id, title, filename, storage_uri, checksum, mime_type, file_size_bytes,
			processing_status, validity_status, uploaded_by
		) VALUES (
			$1,'xa_chu_se','ho_tich_chung_thuc','P4 source','p4.pdf','s3://cas-documents/p4',$2,
			'application/pdf', 12, 'READY', 'VALID', $3
		)`, docID, sha(docID), admin); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO procedure_version_documents (
			procedure_version_id, document_id, procedure_id, xa_id, domain_id, relationship_type, index_status
		) VALUES ($1,$2,$3,'xa_chu_se','ho_tich_chung_thuc','SOURCE','UPLOADED')`,
		versionID, docID, procID); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO document_index_jobs (
			id, document_id, xa_id, request_id, payload_hash, status, procedure_version_id, claim_token, finished_at
		) VALUES ($1,$2,'xa_chu_se',$3,'hash','SUCCEEDED',$4,gen_random_uuid(), now())`,
		jobID, docID, uuid.New(), versionID); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO document_index_generations (
			id, xa_id, document_id, procedure_version_id, job_id, status,
			pipeline_version, extraction_version, ocr_version, chunk_config_hash,
			embedding_model_id, embedding_revision, embedding_checksum, vector_dimension,
			source_sha256, content_sha256, manifest_hash,
			chunk_count, vector_count, published_at
		) VALUES (
			$1,'xa_chu_se',$2,$3,$4,'READY',
			'p4-test','p4-test','p4-test','p4-test',
			'e5-small','rev','',384,
			$5,'','',
			2,2, now()
		)`, genID, docID, versionID, jobID, sha(docID)); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		UPDATE procedure_version_documents
		SET index_status = 'READY', active_generation_id = $3
		WHERE procedure_version_id = $1 AND document_id = $2`, versionID, docID, genID); err != nil {
		t.Fatal(err)
	}
}

func newFixture(t *testing.T, pool *pgxpool.Pool, withV1Active bool) fixture {
	t.Helper()
	ctx := context.Background()
	admin := uuid.MustParse(seedAdmin)
	code := "p4_act_" + uuid.NewString()[:8]
	fx := fixture{
		ProcedureID: uuid.New(),
		V1:          uuid.New(),
		V2:          uuid.New(),
		DocID:       uuid.New(),
		pool:        pool,
		svc:         &activation.Service{Store: &activation.Store{Pool: pool}, XAID: "xa_chu_se"},
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO procedures (id, procedure_code, domain_id, name, xa_id)
		VALUES ($1,$2,'ho_tich_chung_thuc',$2,'xa_chu_se')`, fx.ProcedureID, code); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO procedure_versions (id, procedure_id, version, status, definition, created_by)
		VALUES ($1,$2,'1.0.0',$3,$4::jsonb,$5)`,
		fx.V1, fx.ProcedureID, map[bool]string{true: "ACTIVE", false: "APPROVED"}[withV1Active],
		validDef(code, code, "1.0.0"), admin); err != nil {
		t.Fatal(err)
	}
	if withV1Active {
		if _, err := pool.Exec(ctx, `UPDATE procedures SET active_version_id = $2 WHERE id = $1`, fx.ProcedureID, fx.V1); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO procedure_versions (id, procedure_id, version, status, definition, created_by, approved_by, approved_at)
		VALUES ($1,$2,'2.0.0','APPROVED',$3::jsonb,$4,$4,now())`,
		fx.V2, fx.ProcedureID, validDef(code, code, "2.0.0"), admin); err != nil {
		t.Fatal(err)
	}
	insertReadySource(t, pool, fx.ProcedureID, fx.V2, fx.DocID)
	t.Cleanup(func() { cleanupFixture(pool, fx) })
	return fx
}

func cleanupFixture(pool *pgxpool.Pool, fx fixture) {
	ctx := context.Background()
	var docs []uuid.UUID
	rows, _ := pool.Query(ctx, `SELECT document_id FROM procedure_version_documents WHERE procedure_id = $1`, fx.ProcedureID)
	if rows != nil {
		for rows.Next() {
			var id uuid.UUID
			_ = rows.Scan(&id)
			docs = append(docs, id)
		}
		rows.Close()
	}
	docs = append(docs, fx.DocID)
	_, _ = pool.Exec(ctx, `DELETE FROM session_slot_states WHERE procedure_id = $1`, fx.ProcedureID)
	_, _ = pool.Exec(ctx, `DELETE FROM conversation_sessions WHERE active_procedure_id = $1`, fx.ProcedureID)
	_, _ = pool.Exec(ctx, `DELETE FROM audit_logs WHERE entity_id = $1 OR payload->>'procedure_id' = $1`, fx.ProcedureID.String())
	_, _ = pool.Exec(ctx, `DELETE FROM admin_write_idempotency WHERE action IN ('ACTIVATE_VERSION','CREATE_VERSION') AND response_json->>'procedure_id' = $1`, fx.ProcedureID.String())
	_, _ = pool.Exec(ctx, `UPDATE procedure_version_documents SET active_generation_id = NULL WHERE procedure_id = $1`, fx.ProcedureID)
	_, _ = pool.Exec(ctx, `DELETE FROM knowledge_chunks WHERE procedure_version_id IN (SELECT id FROM procedure_versions WHERE procedure_id = $1)`, fx.ProcedureID)
	_, _ = pool.Exec(ctx, `DELETE FROM document_index_generations WHERE procedure_version_id IN (SELECT id FROM procedure_versions WHERE procedure_id = $1)`, fx.ProcedureID)
	_, _ = pool.Exec(ctx, `DELETE FROM document_index_jobs WHERE procedure_version_id IN (SELECT id FROM procedure_versions WHERE procedure_id = $1)`, fx.ProcedureID)
	_, _ = pool.Exec(ctx, `DELETE FROM procedure_version_documents WHERE procedure_id = $1`, fx.ProcedureID)
	_, _ = pool.Exec(ctx, `UPDATE procedures SET active_version_id = NULL WHERE id = $1`, fx.ProcedureID)
	_, _ = pool.Exec(ctx, `DELETE FROM procedure_versions WHERE procedure_id = $1`, fx.ProcedureID)
	_, _ = pool.Exec(ctx, `DELETE FROM procedures WHERE id = $1`, fx.ProcedureID)
	for _, id := range docs {
		_, _ = pool.Exec(ctx, `DELETE FROM documents WHERE id = $1`, id)
	}
}

func actor() uuid.UUID { return uuid.MustParse(seedAdmin) }

func TestAFirstActivation(t *testing.T) {
	pool := requireP4DB(t)
	fx := newFixture(t, pool, false)
	out, err := fx.svc.Activate(context.Background(), actor(), uuid.New(), fx.ProcedureID, fx.V2, nil)
	if err != nil {
		t.Fatal(err)
	}
	if out.ActiveVersionID != fx.V2 || out.Status != "ACTIVE" {
		t.Fatalf("%+v", out)
	}
	assertOneActive(t, pool, fx.ProcedureID, fx.V2)
	assertAudit(t, pool, fx.ProcedureID, 1)
}

func TestBReplaceExistingVersion(t *testing.T) {
	pool := requireP4DB(t)
	fx := newFixture(t, pool, true)
	out, err := fx.svc.Activate(context.Background(), actor(), uuid.New(), fx.ProcedureID, fx.V2, &fx.V1)
	if err != nil {
		t.Fatal(err)
	}
	if out.PreviousActiveVersionID == nil || *out.PreviousActiveVersionID != fx.V1 {
		t.Fatalf("%+v", out)
	}
	assertOneActive(t, pool, fx.ProcedureID, fx.V2)
	var status, def string
	if err := pool.QueryRow(context.Background(), `SELECT status, definition::text FROM procedure_versions WHERE id = $1`, fx.V1).Scan(&status, &def); err != nil {
		t.Fatal(err)
	}
	if status != "ARCHIVED" || !strings.Contains(def, "1.0.0") {
		t.Fatalf("v1 status=%s def=%s", status, def)
	}
	assertAudit(t, pool, fx.ProcedureID, 1)
}

func TestActivateArchivesLiveActiveWhenPointerIsStale(t *testing.T) {
	pool := requireP4DB(t)
	fx := newFixture(t, pool, true)
	stale := uuid.New()
	admin := actor()
	var code, name string
	if err := pool.QueryRow(context.Background(), `SELECT procedure_code, name FROM procedures WHERE id=$1`, fx.ProcedureID).Scan(&code, &name); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(context.Background(), `
		INSERT INTO procedure_versions (id, procedure_id, version, status, definition, created_by, approved_by, approved_at)
		VALUES ($1,$2,'1.0.0-stale','ARCHIVED',$3::jsonb,$4,$4,now())`,
		stale, fx.ProcedureID, validDef(code, name, "1.0.0-stale"), admin); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(context.Background(), `UPDATE procedures SET active_version_id = $2 WHERE id = $1`, fx.ProcedureID, stale); err != nil {
		t.Fatal(err)
	}
	out, err := fx.svc.Activate(context.Background(), admin, uuid.New(), fx.ProcedureID, fx.V2, &stale)
	if err != nil {
		t.Fatal(err)
	}
	if out.PreviousActiveVersionID == nil || *out.PreviousActiveVersionID != fx.V1 {
		t.Fatalf("previous=%v want live ACTIVE %s", out.PreviousActiveVersionID, fx.V1)
	}
	assertOneActive(t, pool, fx.ProcedureID, fx.V2)
	var v1Status string
	if err := pool.QueryRow(context.Background(), `SELECT status FROM procedure_versions WHERE id=$1`, fx.V1).Scan(&v1Status); err != nil {
		t.Fatal(err)
	}
	if v1Status != "ARCHIVED" {
		t.Fatalf("live ACTIVE not archived: %s", v1Status)
	}
}

func TestCConcurrentActivation(t *testing.T) {
	pool := requireP4DB(t)
	fx := newFixture(t, pool, true)
	v3 := uuid.New()
	admin := actor()
	var code, name string
	if err := pool.QueryRow(context.Background(), `SELECT procedure_code, name FROM procedures WHERE id=$1`, fx.ProcedureID).Scan(&code, &name); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(context.Background(), `
		INSERT INTO procedure_versions (id, procedure_id, version, status, definition, created_by, approved_by, approved_at)
		VALUES ($1,$2,'3.0.0','APPROVED',$3::jsonb,$4,$4,now())`,
		v3, fx.ProcedureID, validDef(code, name, "3.0.0"), admin); err != nil {
		t.Fatal(err)
	}
	insertReadySource(t, pool, fx.ProcedureID, v3, uuid.New())
	var wg sync.WaitGroup
	errs := make(chan error, 2)
	wg.Add(2)
	go func() {
		defer wg.Done()
		_, err := fx.svc.Activate(context.Background(), admin, uuid.New(), fx.ProcedureID, fx.V2, &fx.V1)
		errs <- err
	}()
	go func() {
		defer wg.Done()
		_, err := fx.svc.Activate(context.Background(), admin, uuid.New(), fx.ProcedureID, v3, &fx.V1)
		errs <- err
	}()
	wg.Wait()
	close(errs)
	var okN, staleN int
	for err := range errs {
		if err == nil {
			okN++
			continue
		}
		if errors.Is(err, activation.ErrStale) {
			staleN++
			continue
		}
		t.Fatalf("unexpected %v", err)
	}
	if okN != 1 || staleN != 1 {
		t.Fatalf("ok=%d stale=%d", okN, staleN)
	}
	var n int
	if err := pool.QueryRow(context.Background(), `SELECT count(*) FROM procedure_versions WHERE procedure_id=$1 AND status='ACTIVE'`, fx.ProcedureID).Scan(&n); err != nil {
		t.Fatal(err)
	}
	if n != 1 {
		t.Fatalf("active count=%d", n)
	}
	assertAudit(t, pool, fx.ProcedureID, 1)
}

func TestDFailureRollback(t *testing.T) {
	pool := requireP4DB(t)
	fx := newFixture(t, pool, true)
	missingActor := uuid.New()
	_, err := fx.svc.Activate(context.Background(), missingActor, uuid.New(), fx.ProcedureID, fx.V2, &fx.V1)
	if err == nil {
		t.Fatal("expected FK failure")
	}
	assertOneActive(t, pool, fx.ProcedureID, fx.V1)
	var v2status string
	if err := pool.QueryRow(context.Background(), `SELECT status FROM procedure_versions WHERE id=$1`, fx.V2).Scan(&v2status); err != nil {
		t.Fatal(err)
	}
	if v2status != "APPROVED" {
		t.Fatalf("v2=%s", v2status)
	}
	assertAudit(t, pool, fx.ProcedureID, 0)
}

func TestEIdempotencyReplay(t *testing.T) {
	pool := requireP4DB(t)
	fx := newFixture(t, pool, true)
	rid := uuid.New()
	first, err := fx.svc.Activate(context.Background(), actor(), rid, fx.ProcedureID, fx.V2, &fx.V1)
	if err != nil {
		t.Fatal(err)
	}
	second, err := fx.svc.Activate(context.Background(), actor(), rid, fx.ProcedureID, fx.V2, &fx.V1)
	if err != nil {
		t.Fatal(err)
	}
	if !second.Replayed || second.ActiveVersionID != first.ActiveVersionID {
		t.Fatalf("%+v", second)
	}
	assertAudit(t, pool, fx.ProcedureID, 1)
	_, err = fx.svc.Activate(context.Background(), actor(), rid, fx.ProcedureID, fx.V2, nil)
	if !errors.Is(err, activation.ErrIdempotency) {
		t.Fatalf("want idempotency conflict got %v", err)
	}
}

func TestFSessionPin(t *testing.T) {
	pool := requireP4DB(t)
	fx := newFixture(t, pool, true)
	ctx := context.Background()
	sessionID := uuid.New()
	if _, err := pool.Exec(ctx, `
		INSERT INTO conversation_sessions (id, xa_id, guest_token, active_procedure_id, active_procedure_version_id, status)
		VALUES ($1,'xa_chu_se',$2,$3,$4,'OPEN')`, sessionID, "p4-"+uuid.NewString(), fx.ProcedureID, fx.V1); err != nil {
		t.Fatal(err)
	}
	if _, err := fx.svc.Activate(ctx, actor(), uuid.New(), fx.ProcedureID, fx.V2, &fx.V1); err != nil {
		t.Fatal(err)
	}
	var pinned uuid.UUID
	if err := pool.QueryRow(ctx, `SELECT active_procedure_version_id FROM conversation_sessions WHERE id=$1`, sessionID).Scan(&pinned); err != nil {
		t.Fatal(err)
	}
	if pinned != fx.V1 {
		t.Fatalf("pin jumped to %s", pinned)
	}
	tx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer tx.Rollback(ctx)
	snap, err := (&repository.SessionRepo{Pool: pool}).LockTurn(ctx, tx, sessionID, []string{"ho_tich_chung_thuc"})
	if err != nil {
		t.Fatal(err)
	}
	if snap.Pinned == nil || snap.Pinned.VersionID != fx.V1 {
		t.Fatal("pinned snapshot lost V1")
	}
	foundV2 := false
	for _, row := range snap.Catalog {
		if row.ProcedureID == fx.ProcedureID && row.VersionID == fx.V2 {
			foundV2 = true
		}
		if row.ProcedureID == fx.ProcedureID && row.VersionID == fx.V1 {
			t.Fatal("catalog still listed archived V1 as ACTIVE")
		}
	}
	if !foundV2 {
		t.Fatal("new catalog missing V2")
	}
}

func TestGRaceUnlink(t *testing.T) {
	pool := requireP4DB(t)
	fx := newFixture(t, pool, true)
	blocker, err := pool.Begin(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := blocker.Exec(context.Background(), `
		SELECT document_id FROM procedure_version_documents
		WHERE procedure_version_id=$1 FOR UPDATE`, fx.V2); err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() {
		_, err := fx.svc.Activate(context.Background(), actor(), uuid.New(), fx.ProcedureID, fx.V2, &fx.V1)
		done <- err
	}()
	if _, err := blocker.Exec(context.Background(), `
		UPDATE procedure_version_documents
		SET unlinked_at = now(), active_generation_id = NULL, index_status = 'UPLOADED'
		WHERE procedure_version_id=$1`, fx.V2); err != nil {
		t.Fatal(err)
	}
	if err := blocker.Commit(context.Background()); err != nil {
		t.Fatal(err)
	}
	err = <-done
	if err == nil {
		t.Fatal("activated after unlink")
	}
	assertOneActive(t, pool, fx.ProcedureID, fx.V1)
}

func TestHMigrationCompatibility(t *testing.T) {
	pool := requireP4DB(t)
	var activatedAtExists bool
	if err := pool.QueryRow(context.Background(), `
		SELECT EXISTS (
			SELECT 1 FROM information_schema.columns
			WHERE table_name='procedure_versions' AND column_name='activated_at'
		)`).Scan(&activatedAtExists); err != nil {
		t.Fatal(err)
	}
	if !activatedAtExists {
		t.Fatal("000017 activated_at missing")
	}
	var n int
	if err := pool.QueryRow(context.Background(), `
		SELECT count(*) FROM procedure_versions pv
		JOIN procedures p ON p.id = pv.procedure_id
		WHERE pv.status='ACTIVE' AND p.active_version_id = pv.id`).Scan(&n); err != nil {
		t.Fatal(err)
	}
	if n < 1 {
		t.Fatal("seed ACTIVE rows were lost")
	}
}

func TestHTTPAuthAndEligibility(t *testing.T) {
	pool := requireP4DB(t)
	fx := newFixture(t, pool, true)
	eng, adminTok, citizenTok := testEngine(t, pool)
	path := fmt.Sprintf("/api/v1/admin/procedures/%s/versions/%s/activate", fx.ProcedureID, fx.V2)
	code, _ := callJSON(t, eng, http.MethodPost, path, map[string]any{"expected_active_version_id": fx.V1.String()}, citizenTok, uuid.New())
	if code != http.StatusForbidden {
		t.Fatalf("citizen %d", code)
	}
	code, _ = callJSON(t, eng, http.MethodPost, path, map[string]any{"expected_active_version_id": fx.V1.String()}, "", uuid.New())
	if code != http.StatusUnauthorized {
		t.Fatalf("guest %d", code)
	}
	if _, err := pool.Exec(context.Background(), `
		UPDATE procedure_version_documents SET active_generation_id=NULL, index_status='READY'
		WHERE procedure_version_id=$1`, fx.V2); err != nil {
		t.Fatal(err)
	}
	code, env := callJSON(t, eng, http.MethodPost, path, map[string]any{"expected_active_version_id": fx.V1.String()}, adminTok, uuid.New())
	if code != http.StatusUnprocessableEntity {
		t.Fatalf("mock ready %d %v", code, env)
	}
	errObj, _ := env["error"].(map[string]any)
	if fmt.Sprint(errObj["code"]) != "ACTIVE_GENERATION_MISSING" && fmt.Sprint(errObj["code"]) != "MOCK_INDEX_NOT_PUBLISHED" {
		t.Fatalf("code=%v", errObj)
	}
	listCode, listEnv := callJSON(t, eng, http.MethodGet, "/api/v1/admin/procedures", nil, adminTok, uuid.Nil)
	if listCode != http.StatusOK {
		t.Fatalf("list %d %v", listCode, listEnv)
	}
}

func TestHTTPActivateSuccessAndPublicCatalog(t *testing.T) {
	pool := requireP4DB(t)
	fx := newFixture(t, pool, true)
	eng, adminTok, _ := testEngine(t, pool)
	path := fmt.Sprintf("/api/v1/admin/procedures/%s/versions/%s/activate", fx.ProcedureID, fx.V2)
	rid := uuid.New()
	code, env := callJSON(t, eng, http.MethodPost, path, map[string]any{"expected_active_version_id": fx.V1.String()}, adminTok, rid)
	if code != http.StatusOK {
		t.Fatalf("activate %d %v", code, env)
	}
	code, replay := callJSON(t, eng, http.MethodPost, path, map[string]any{"expected_active_version_id": fx.V1.String()}, adminTok, rid)
	if code != http.StatusOK {
		t.Fatalf("replay %d %v", code, replay)
	}
	data, _ := replay["data"].(map[string]any)
	if data["replayed"] != true {
		t.Fatalf("replayed=%v", data)
	}
	pub := fmt.Sprintf("/api/v1/procedures/%s/active-version", fx.ProcedureID)
	code, pubEnv := callJSON(t, eng, http.MethodGet, pub, nil, "", uuid.Nil)
	if code != http.StatusOK {
		t.Fatalf("public %d %v", code, pubEnv)
	}
	pubData, _ := pubEnv["data"].(map[string]any)
	if fmt.Sprint(pubData["id"]) != fx.V2.String() {
		t.Fatalf("public active=%v", pubData["id"])
	}
	histPath := fmt.Sprintf("/api/v1/admin/procedures/%s/activation-history", fx.ProcedureID)
	code, hist := callJSON(t, eng, http.MethodGet, histPath, nil, adminTok, uuid.Nil)
	if code != http.StatusOK {
		t.Fatalf("history %d %v", code, hist)
	}
}

func callJSON(t *testing.T, eng http.Handler, method, path string, body any, token string, requestID uuid.UUID) (int, map[string]any) {
	t.Helper()
	var buf bytes.Buffer
	if body != nil {
		if err := json.NewEncoder(&buf).Encode(body); err != nil {
			t.Fatal(err)
		}
	}
	req := httptest.NewRequest(method, path, &buf)
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	if requestID != uuid.Nil {
		req.Header.Set("X-Request-ID", requestID.String())
	}
	w := httptest.NewRecorder()
	eng.ServeHTTP(w, req)
	var env map[string]any
	_ = json.Unmarshal(w.Body.Bytes(), &env)
	return w.Code, env
}

func assertOneActive(t *testing.T, pool *pgxpool.Pool, procedureID, want uuid.UUID) {
	t.Helper()
	var n int
	var pointer uuid.UUID
	var status string
	if err := pool.QueryRow(context.Background(), `
		SELECT count(*) FROM procedure_versions WHERE procedure_id=$1 AND status='ACTIVE'`, procedureID).Scan(&n); err != nil {
		t.Fatal(err)
	}
	if n != 1 {
		t.Fatalf("active count=%d", n)
	}
	if err := pool.QueryRow(context.Background(), `
		SELECT p.active_version_id, pv.status
		FROM procedures p JOIN procedure_versions pv ON pv.id = p.active_version_id
		WHERE p.id=$1`, procedureID).Scan(&pointer, &status); err != nil {
		t.Fatal(err)
	}
	if pointer != want || status != "ACTIVE" {
		t.Fatalf("pointer=%s status=%s", pointer, status)
	}
}

func assertAudit(t *testing.T, pool *pgxpool.Pool, procedureID uuid.UUID, want int) {
	t.Helper()
	var n int
	if err := pool.QueryRow(context.Background(), `
		SELECT count(*) FROM audit_logs
		WHERE action='PROCEDURE_VERSION_ACTIVATED' AND entity_id=$1`, procedureID.String()).Scan(&n); err != nil {
		t.Fatal(err)
	}
	if n != want {
		t.Fatalf("audit=%d want=%d", n, want)
	}
}
