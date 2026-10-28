//go:build integration

package activation_test

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"testing"
	"time"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/activation"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/auth"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/config"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/drafting"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/httpserver"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/repository"
	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
)

// engineForCommune builds the HTTP server scoped to xaID, so a fresh commune
// stands in for an empty database without touching other fixtures.
func engineForCommune(t *testing.T, pool *pgxpool.Pool, xaID string) (http.Handler, string, string) {
	t.Helper()
	cfg := config.Config{
		Env: "local", XAID: xaID, LogLevel: "error",
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
	return httpserver.New(slog.New(slog.NewTextHandler(io.Discard, nil)), pool, cfg), adminTok, citizenTok
}

func dataOf(env map[string]any) map[string]any {
	d, _ := env["data"].(map[string]any)
	return d
}

func errCode(env map[string]any) string {
	e, _ := env["error"].(map[string]any)
	s, _ := e["code"].(string)
	return s
}

func TestCreateFirstProcedureInEmptyCommune(t *testing.T) {
	pool := requireP4DB(t)
	ctx := context.Background()
	xa := "xa_p4_new_" + uuid.NewString()[:8]
	if _, err := pool.Exec(ctx, `INSERT INTO communes (id, name) VALUES ($1,$1)`, xa); err != nil {
		t.Fatal(err)
	}
	eng, adminTok, citizenTok := engineForCommune(t, pool, xa)
	const path = "/api/v1/admin/procedures"

	code, env := callJSON(t, eng, http.MethodGet, path, nil, adminTok, uuid.Nil)
	if code != http.StatusOK || dataOf(env)["count"].(float64) != 0 {
		t.Fatalf("empty commune list %d %v", code, env)
	}

	procCode := "dk_moi_" + uuid.NewString()[:8]
	body := map[string]any{"procedure_code": procCode, "name": "Đăng ký thủ tục mới", "domain_id": "ho_tich_chung_thuc"}

	if code, _ := callJSON(t, eng, http.MethodPost, path, body, "", uuid.New()); code != http.StatusUnauthorized {
		t.Fatalf("no token want 401 got %d", code)
	}
	if code, _ := callJSON(t, eng, http.MethodPost, path, body, citizenTok, uuid.New()); code != http.StatusForbidden {
		t.Fatalf("citizen want 403 got %d", code)
	}
	if code, _ := callJSON(t, eng, http.MethodPost, path, body, adminTok, uuid.Nil); code != http.StatusUnprocessableEntity {
		t.Fatalf("missing X-Request-ID want 422 got %d", code)
	}
	for name, bad := range map[string]map[string]any{
		"uppercase code": {"procedure_code": "DK_Moi", "name": "x", "domain_id": "ho_tich_chung_thuc"},
		"short code":     {"procedure_code": "a", "name": "x", "domain_id": "ho_tich_chung_thuc"},
		"empty name":     {"procedure_code": procCode, "name": "  ", "domain_id": "ho_tich_chung_thuc"},
		"unknown domain": {"procedure_code": procCode, "name": "x", "domain_id": "khong_ton_tai"},
		"unknown field":  {"procedure_code": procCode, "name": "x", "domain_id": "ho_tich_chung_thuc", "version": "1.0.0"},
	} {
		if code, env := callJSON(t, eng, http.MethodPost, path, bad, adminTok, uuid.New()); code != http.StatusBadRequest || errCode(env) != "VALIDATION" {
			t.Fatalf("%s want 400 VALIDATION got %d %v", name, code, env)
		}
	}

	rid := uuid.New()
	code, env = callJSON(t, eng, http.MethodPost, path, body, adminTok, rid)
	if code != http.StatusCreated {
		t.Fatalf("create %d %v", code, env)
	}
	created := dataOf(env)
	procID := uuid.MustParse(created["id"].(string))
	sourceID := uuid.MustParse(created["source_version_id"].(string))
	if created["xa_id"] != xa || created["source_version"] != activation.SourceVersionLabel || created["procedure_code"] != procCode {
		t.Fatalf("create payload %v", created)
	}

	var activePtr *uuid.UUID
	var status, def string
	if err := pool.QueryRow(ctx, `SELECT active_version_id FROM procedures WHERE id=$1`, procID).Scan(&activePtr); err != nil {
		t.Fatal(err)
	}
	if activePtr != nil {
		t.Fatalf("new procedure must not point to an active version: %v", activePtr)
	}
	var versions, actives int
	if err := pool.QueryRow(ctx, `
		SELECT count(*), count(*) FILTER (WHERE status='ACTIVE') FROM procedure_versions WHERE procedure_id=$1`, procID).
		Scan(&versions, &actives); err != nil {
		t.Fatal(err)
	}
	if versions != 1 || actives != 0 {
		t.Fatalf("versions=%d actives=%d", versions, actives)
	}
	if err := pool.QueryRow(ctx, `SELECT status, definition::text FROM procedure_versions WHERE id=$1`, sourceID).Scan(&status, &def); err != nil {
		t.Fatal(err)
	}
	if status != activation.StatusSource || def != "{}" {
		t.Fatalf("source version status=%s definition=%s", status, def)
	}
	var audits int
	if err := pool.QueryRow(ctx, `
		SELECT count(*) FROM audit_logs WHERE action=$1 AND entity_id=$2`, activation.AuditCreateProcedure, procID.String()).Scan(&audits); err != nil {
		t.Fatal(err)
	}
	if audits != 1 {
		t.Fatalf("audit rows=%d", audits)
	}

	code, env = callJSON(t, eng, http.MethodPost, path, body, adminTok, rid)
	if code != http.StatusOK || dataOf(env)["replayed"] != true || dataOf(env)["id"] != procID.String() {
		t.Fatalf("replay %d %v", code, env)
	}
	changed := map[string]any{"procedure_code": procCode, "name": "Tên khác", "domain_id": "ho_tich_chung_thuc"}
	if code, env := callJSON(t, eng, http.MethodPost, path, changed, adminTok, rid); code != http.StatusConflict || errCode(env) != "IDEMPOTENCY_CONFLICT" {
		t.Fatalf("same request id different body %d %v", code, env)
	}
	if code, env := callJSON(t, eng, http.MethodPost, path, body, adminTok, uuid.New()); code != http.StatusConflict || errCode(env) != "PROCEDURE_CODE_EXISTS" {
		t.Fatalf("duplicate code %d %v", code, env)
	}

	code, env = callJSON(t, eng, http.MethodGet, path, nil, adminTok, uuid.Nil)
	items, _ := dataOf(env)["items"].([]any)
	if code != http.StatusOK || len(items) != 1 {
		t.Fatalf("list after create %d %v", code, env)
	}
	if row := items[0].(map[string]any); row["active_version_id"] != nil || row["version_count"].(float64) != 1 {
		t.Fatalf("list row %v", row)
	}

	code, env = callJSON(t, eng, http.MethodGet, path+"/"+procID.String()+"/versions", nil, adminTok, uuid.Nil)
	vitems, _ := dataOf(env)["items"].([]any)
	if code != http.StatusOK || len(vitems) != 1 {
		t.Fatalf("versions %d %v", code, env)
	}
	elig := vitems[0].(map[string]any)["eligibility"].(map[string]any)
	if elig["can_activate"] != false {
		t.Fatalf("SOURCE must not be activatable: %v", elig)
	}

	code, env = callJSON(t, eng, http.MethodPost, path+"/"+procID.String()+"/versions/"+sourceID.String()+"/activate",
		map[string]any{"expected_active_version_id": nil}, adminTok, uuid.New())
	if code == http.StatusOK {
		t.Fatalf("SOURCE activate must fail, got %d %v", code, env)
	}
	if err := pool.QueryRow(ctx, `
		SELECT count(*) FILTER (WHERE status='ACTIVE') FROM procedure_versions WHERE procedure_id=$1`, procID).Scan(&actives); err != nil {
		t.Fatal(err)
	}
	if actives != 0 {
		t.Fatalf("activate on SOURCE produced %d ACTIVE rows", actives)
	}

	copyBody := map[string]any{"copy_from_version_id": sourceID, "confirm": true}
	if code, env := callJSON(t, eng, http.MethodPost, path+"/"+procID.String()+"/versions", copyBody, adminTok, uuid.New()); code != http.StatusBadRequest {
		t.Fatalf("copy from SOURCE want 400 got %d %v", code, env)
	}
	versionBody := map[string]any{"definition": json.RawMessage(validDef(procCode, "Đăng ký thủ tục mới", "1.0.0")), "confirm": true}
	code, env = callJSON(t, eng, http.MethodPost, path+"/"+procID.String()+"/versions", versionBody, adminTok, uuid.New())
	if code != http.StatusCreated || dataOf(env)["version"] != "1.0.0" || dataOf(env)["status"] != "APPROVED" {
		t.Fatalf("first real version %d %v", code, env)
	}
}

func TestSourceVersionAnchorsFirstPDFForAIDraft(t *testing.T) {
	pool := requireP4DB(t)
	ctx := context.Background()
	svc := &activation.Service{Store: &activation.Store{Pool: pool}, XAID: "xa_chu_se"}
	created, err := svc.CreateProcedure(ctx, uuid.MustParse(seedAdmin), uuid.New(), activation.CreateProcedureRequest{
		ProcedureCode: "p4_src_" + uuid.NewString()[:8], Name: "Nguồn PDF đầu tiên", DomainID: "ho_tich_chung_thuc",
	})
	if err != nil {
		t.Fatal(err)
	}

	targets, err := (&repository.IndexRepo{Pool: pool}).ListTargets(ctx, "xa_chu_se")
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, tg := range targets {
		if tg.ProcedureVersionID == created.SourceVersionID {
			found = tg.Status == activation.StatusSource
		}
	}
	if !found {
		t.Fatalf("SOURCE version must be a PDF link target")
	}

	docID := uuid.New()
	insertReadySource(t, pool, created.ID, created.SourceVersionID, docID)
	var genID uuid.UUID
	if err := pool.QueryRow(ctx, `SELECT id FROM document_index_generations WHERE document_id=$1`, docID).Scan(&genID); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO knowledge_chunks (
			xa_id, procedure_id, procedure_version_id, document_id, chunk_index, content,
			generation_id, page_start, page_end, text_sha256, token_count, extraction_source
		) VALUES ('xa_chu_se',$1,$2,$3,0,'Điều 1. Hồ sơ gồm tờ khai.',$4,10,12,$5,6,'native')`,
		created.ID, created.SourceVersionID, docID, genID, sha(docID)); err != nil {
		t.Fatal(err)
	}

	sources, err := (&drafting.SourceStore{Pool: pool}).ListReady(ctx, "xa_chu_se", created.ID)
	if err != nil {
		t.Fatal(err)
	}
	if len(sources) != 1 || sources[0].DocumentID != docID || sources[0].VersionStatus != activation.StatusSource {
		t.Fatalf("AI draft sources %+v", sources)
	}

	summary, err := svc.GetVersion(ctx, created.ID, created.SourceVersionID)
	if err != nil {
		t.Fatal(err)
	}
	if summary.Eligibility.CanActivate || len(summary.Eligibility.Reasons) != 1 || summary.Eligibility.Reasons[0] != "SOURCE_VERSION" {
		t.Fatalf("SOURCE with READY PDF must stay non-activatable: %+v", summary.Eligibility)
	}
}
