//go:build integration

package index_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"testing"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/index"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/repository"
)

func requireP4BTestDB(t *testing.T) *pgxpool.Pool {
	t.Helper()
	if os.Getenv("CAS_INTEGRATION") != "1" {
		t.Skip("CAS_INTEGRATION=1 required only after isolation pin")
	}
	for _, name := range []string{"PGHOST", "PGHOSTADDR", "PGPORT", "PGDATABASE", "PGSERVICE"} {
		if strings.TrimSpace(os.Getenv(name)) != "" {
			t.Skipf("libpq override %s set; p4b-test isolation not pinned", name)
		}
	}
	raw := strings.TrimSpace(os.Getenv("DATABASE_URL"))
	if raw == "" {
		t.Skip("DATABASE_URL required; refusing implicit 127.0.0.1:5432 fallback")
	}
	if strings.Contains(raw, ":5432") || strings.HasSuffix(raw, "/citizen_assistance") || strings.Contains(raw, "/citizen_assistance?") {
		t.Skip("requires citizen_assistance_p4b_test; refusing CI/shared DATABASE_URL")
	}
	cfg, err := pgxpool.ParseConfig(raw)
	if err != nil {
		t.Fatal(err)
	}
	host := cfg.ConnConfig.Host
	port := cfg.ConnConfig.Port
	db := cfg.ConnConfig.Database
	if parsed, err := url.Parse(raw); err == nil && parsed.Port() != "" {
		if n, convErr := strconv.Atoi(parsed.Port()); convErr == nil {
			port = uint16(n)
		}
		if parsed.Hostname() != "" {
			host = parsed.Hostname()
		}
	}
	if (host != "127.0.0.1" && host != "localhost") || port != 5434 || db != "citizen_assistance_p4b_test" {
		t.Skipf("requires p4b-test target, got host=%s port=%d db=%s", host, port, db)
	}
	ctx := context.Background()
	pool, err := pgxpool.NewWithConfig(ctx, cfg)
	if err != nil {
		t.Skipf("p4b-test postgres not reachable: %v", err)
	}
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		t.Skip("p4b-test postgres not reachable")
	}
	var name string
	if err := pool.QueryRow(ctx, "SELECT current_database()").Scan(&name); err != nil {
		pool.Close()
		t.Skip("p4b-test postgres not reachable")
	}
	if name != "citizen_assistance_p4b_test" {
		pool.Close()
		t.Skipf("connected database %s", name)
	}
	t.Cleanup(pool.Close)
	return pool
}

type g3Worker struct {
	pool        *pgxpool.Pool
	procedureID string
	mode        string
	claim       uuid.UUID
	jobID       uuid.UUID
	genID       uuid.UUID
	manifest    string
}

func (w *g3Worker) Index(ctx context.Context, req index.Request) (index.Response, error) {
	w.claim = req.ClaimToken
	w.jobID = req.JobID
	w.genID = req.GenerationID
	if req.SchemaVersion != index.SchemaVersionV2 || req.GenerationID == uuid.Nil {
		return index.Response{}, context.Canceled
	}
	text := "g3 finish fixture xin chao"
	sum := sha256.Sum256([]byte(text))
	textSHA := hex.EncodeToString(sum[:])
	pageStart, pageEnd := 10, 10
	line := fmt.Sprintf("0|%d|%d|%s|2|native", pageStart, pageEnd, textSHA)
	manifestSum := sha256.Sum256([]byte(line))
	manifest := hex.EncodeToString(manifestSum[:])
	if w.mode == "mismatch" {
		manifest = hex.EncodeToString(sha256.New().Sum(nil))
	}
	w.manifest = manifest
	source := req.Checksum
	if _, err := w.pool.Exec(ctx, `
		UPDATE document_index_generations
		SET manifest_hash = $2, chunk_count = 1, vector_count = 1, page_count = 1,
			native_page_count = 1, ocr_page_count = 0, content_sha256 = $3,
			pipeline_version = 'p4b.1', source_sha256 = $4
		WHERE id = $1 AND status = 'STAGING'`, req.GenerationID, manifest, textSHA, source); err != nil {
		return index.Response{}, err
	}
	if _, err := w.pool.Exec(ctx, `
		INSERT INTO knowledge_chunks (
			id, xa_id, procedure_id, procedure_version_id, document_id, chunk_index, content,
			generation_id, page_start, page_end, text_sha256, token_count, extraction_source
		) VALUES ($1,$2,$3,$4,$5,0,$6,$7,$8,$9,$10,2,'native')`,
		uuid.New(), req.XAID, w.procedureID, req.ProcedureVersionID, req.DocumentID, text, req.GenerationID, pageStart, pageEnd, textSHA); err != nil {
		return index.Response{}, err
	}
	code := ""
	outcome := index.OutcomeReady
	if w.mode == "fail" {
		outcome = index.OutcomeFailed
		code = "qdrant_failed"
	}
	var errPtr *string
	if code != "" {
		errPtr = &code
	}
	chunks, vectors := 1, 1
	if w.mode == "mismatch" {
		chunks, vectors = 1, 1
	}
	return index.Response{
		SchemaVersion: index.SchemaVersionV2, DocumentID: req.DocumentID, ProcedureVersionID: req.ProcedureVersionID,
		XAID: req.XAID, JobID: req.JobID, GenerationID: req.GenerationID, Outcome: outcome, ErrorCode: errPtr,
		SourceSHA256: source, ContentSHA256: textSHA, PagesProcessed: 1, NativePages: 1,
		ChunkCount: chunks, VectorCount: vectors, ManifestHash: manifest, PipelineVersion: index.PipelineVersion,
		VectorDimension: 384,
	}, nil
}

func pinG2(t *testing.T, pool *pgxpool.Pool, versionID, procedureID, docID, genID, userID string) uuid.UUID {
	t.Helper()
	sessionID := uuid.New()
	messageID := uuid.New()
	ctx := context.Background()
	var chunkID uuid.UUID
	if err := pool.QueryRow(ctx, `SELECT id FROM knowledge_chunks WHERE generation_id = $1 LIMIT 1`, genID).Scan(&chunkID); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO conversation_sessions (id, user_id, guest_token, xa_id, active_procedure_id, active_procedure_version_id, status)
		VALUES ($1,$2,$3,'xa_chu_se',$4,$5,'OPEN')`,
		sessionID, userID, "g3final-"+uuid.NewString(), procedureID, versionID); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO session_source_pin_sets (session_id, procedure_version_id, procedure_id, xa_id, materialized)
		VALUES ($1,$2,$3,'xa_chu_se', true)`, sessionID, versionID, procedureID); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO session_source_pins (
			session_id, xa_id, procedure_id, procedure_version_id, document_id, generation_id, document_title
		) VALUES ($1,'xa_chu_se',$2,$3,$4,$5,'G3 finish G2 pin')`,
		sessionID, procedureID, versionID, docID, genID); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO conversation_messages (id, session_id, request_id, role, content, action)
		VALUES ($1,$2,$3,'ASSISTANT','pinned G2','PROVIDE_FINAL_GUIDANCE')`,
		messageID, sessionID, uuid.New()); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO message_citations (
			message_id, knowledge_chunk_id, document_id, title, generation_id, procedure_version_id,
			evidence_status, availability, excerpt, page_start, page_end
		) VALUES ($1,$2,$3,'G3 finish G2 pin',$4,$5,'related','available','g3 finish fixture xin chao',10,10)`,
		messageID, chunkID, docID, genID, versionID); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_, _ = pool.Exec(context.Background(), `DELETE FROM message_citations WHERE message_id = $1`, messageID)
		_, _ = pool.Exec(context.Background(), `DELETE FROM conversation_messages WHERE session_id = $1`, sessionID)
		_, _ = pool.Exec(context.Background(), `DELETE FROM session_source_pins WHERE session_id = $1`, sessionID)
		_, _ = pool.Exec(context.Background(), `DELETE FROM session_source_pin_sets WHERE session_id = $1`, sessionID)
		_, _ = pool.Exec(context.Background(), `DELETE FROM conversation_sessions WHERE id = $1`, sessionID)
	})
	return sessionID
}

func TestG3FinishP4B_ReadySupersedePinDuplicateAndKeepReady(t *testing.T) {
	pool := requireP4BTestDB(t)
	versionID := insertVersion(t, pool, "xa_chu_se", "ho_tich_chung_thuc", "g3_finish_"+uuid.NewString()[:8])
	t.Cleanup(func() { cleanupVersion(pool, versionID) })
	var procedureID, userID string
	if err := pool.QueryRow(context.Background(), `SELECT procedure_id FROM procedure_versions WHERE id = $1`, versionID).Scan(&procedureID); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(context.Background(), `SELECT id FROM users WHERE email = 'admin@chuse.vn'`).Scan(&userID); err != nil {
		t.Fatal(err)
	}
	worker := &g3Worker{pool: pool, procedureID: procedureID, mode: "ok"}
	eng, admin := pipelineEngine(t, pool, worker)
	docID := uploadDoc(t, eng, admin, "%PDF-1.4\n%g3-finish-"+uuid.NewString()+"\n")
	t.Cleanup(func() { cleanupDoc(pool, docID) })
	linked := call(t, eng, http.MethodPost, "/api/v1/admin/documents/"+docID+"/links", admin, uuid.New(), map[string]string{
		"procedure_version_id": versionID, "relationship_type": "SOURCE", "page_range": "10-21",
	})
	if linked.Code != http.StatusCreated {
		t.Fatalf("link %d %s", linked.Code, linked.Body.String())
	}
	first := call(t, eng, http.MethodPost, "/api/v1/admin/documents/"+docID+"/links/"+versionID+"/index", admin, uuid.New(), nil)
	if first.Code != http.StatusOK || !bytesContains(first.Body.String(), `"link_status":"READY"`) {
		t.Fatalf("g2 finish %d %s", first.Code, first.Body.String())
	}
	g2 := worker.genID
	sessionID := pinG2(t, pool, versionID, procedureID, docID, g2.String(), userID)

	worker.mode = "ok"
	second := call(t, eng, http.MethodPost, "/api/v1/admin/documents/"+docID+"/links/"+versionID+"/index/reindex", admin, uuid.New(), nil)
	if second.Code != http.StatusOK || !bytesContains(second.Body.String(), `"link_status":"READY"`) {
		t.Fatalf("g3 finish %d %s", second.Code, second.Body.String())
	}
	g3 := worker.genID
	var ready, superseded int
	var active, g2Status, g3Status, pinGen string
	if err := pool.QueryRow(context.Background(), `
		SELECT count(*) FILTER (WHERE status = 'READY'), count(*) FILTER (WHERE status = 'SUPERSEDED')
		FROM document_index_generations WHERE document_id = $1`, docID).Scan(&ready, &superseded); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(context.Background(), `
		SELECT active_generation_id::text FROM procedure_version_documents
		WHERE document_id = $1 AND procedure_version_id = $2`, docID, versionID).Scan(&active); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(context.Background(), `SELECT status FROM document_index_generations WHERE id = $1`, g2).Scan(&g2Status); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(context.Background(), `SELECT status FROM document_index_generations WHERE id = $1`, g3).Scan(&g3Status); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(context.Background(), `SELECT generation_id::text FROM session_source_pins WHERE session_id = $1`, sessionID).Scan(&pinGen); err != nil {
		t.Fatal(err)
	}
	if ready != 1 || superseded < 1 || active != g3.String() || g2Status != "SUPERSEDED" || g3Status != "READY" || pinGen != g2.String() {
		t.Fatalf("ready=%d superseded=%d active=%s g2=%s g3=%s pin=%s", ready, superseded, active, g2Status, g3Status, pinGen)
	}

	store := &repository.IndexRepo{Pool: pool}
	dup, err := store.Finish(context.Background(), index.FinishInput{
		JobID: worker.jobID, ClaimToken: worker.claim,
		DocumentID: uuid.MustParse(docID), ProcedureVersionID: uuid.MustParse(versionID), XAID: "xa_chu_se",
		ActorUserID: uuid.MustParse(userID), RequestID: uuid.New(), Outcome: index.OutcomeReady,
		GenerationID: g3, ManifestHash: worker.manifest, ChunkCount: 1, VectorCount: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if dup.Applied {
		t.Fatal("duplicate Finish after SUCCEEDED must not apply")
	}
	if err := pool.QueryRow(context.Background(), `SELECT status FROM document_index_generations WHERE id = $1`, g3).Scan(&g3Status); err != nil {
		t.Fatal(err)
	}
	if g3Status != "READY" {
		t.Fatalf("duplicate mutated g3 %s", g3Status)
	}

	worker.mode = "fail"
	failed := call(t, eng, http.MethodPost, "/api/v1/admin/documents/"+docID+"/links/"+versionID+"/index/reindex", admin, uuid.New(), nil)
	if failed.Code != http.StatusOK || !bytesContains(failed.Body.String(), `"link_status":"READY"`) {
		t.Fatalf("failed reindex %d %s", failed.Code, failed.Body.String())
	}
	var still string
	var reindexErr *string
	if err := pool.QueryRow(context.Background(), `
		SELECT active_generation_id::text, reindex_error_code FROM procedure_version_documents
		WHERE document_id = $1 AND procedure_version_id = $2`, docID, versionID).Scan(&still, &reindexErr); err != nil {
		t.Fatal(err)
	}
	if still != g3.String() || reindexErr == nil || *reindexErr == "" {
		t.Fatalf("keepReady lost active=%s err=%v", still, reindexErr)
	}
}

func TestG3FinishP4B_MismatchDoesNotPublish(t *testing.T) {
	pool := requireP4BTestDB(t)
	versionID := insertVersion(t, pool, "xa_chu_se", "ho_tich_chung_thuc", "g3_mismatch_"+uuid.NewString()[:8])
	t.Cleanup(func() { cleanupVersion(pool, versionID) })
	var procedureID string
	if err := pool.QueryRow(context.Background(), `SELECT procedure_id FROM procedure_versions WHERE id = $1`, versionID).Scan(&procedureID); err != nil {
		t.Fatal(err)
	}
	worker := &g3Worker{pool: pool, procedureID: procedureID, mode: "mismatch"}
	eng, admin := pipelineEngine(t, pool, worker)
	docID := uploadDoc(t, eng, admin, "%PDF-1.4\n%g3-mismatch-"+uuid.NewString()+"\n")
	t.Cleanup(func() { cleanupDoc(pool, docID) })
	linkDoc(t, eng, admin, docID, versionID)
	res := call(t, eng, http.MethodPost, "/api/v1/admin/documents/"+docID+"/links/"+versionID+"/index", admin, uuid.New(), nil)
	if res.Code != http.StatusOK || !bytesContains(res.Body.String(), `"error_code":"publish_mismatch"`) {
		t.Fatalf("mismatch %d %s", res.Code, res.Body.String())
	}
	var status string
	var active *string
	if err := pool.QueryRow(context.Background(), `SELECT status FROM document_index_generations WHERE id = $1`, worker.genID).Scan(&status); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(context.Background(), `
		SELECT active_generation_id::text FROM procedure_version_documents
		WHERE document_id = $1 AND procedure_version_id = $2`, docID, versionID).Scan(&active); err != nil {
		t.Fatal(err)
	}
	if status == "READY" || active != nil {
		t.Fatalf("mismatch published status=%s active=%v", status, active)
	}
}
