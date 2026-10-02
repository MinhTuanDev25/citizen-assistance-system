//go:build integration

package index_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/golang-migrate/migrate/v4"
	_ "github.com/golang-migrate/migrate/v4/database/postgres"
	_ "github.com/golang-migrate/migrate/v4/source/file"
	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestUpgradeFromV10RepairsUnlinkedReadyDocument(t *testing.T) {
	name, dsn := scratchDB(t)
	migrateTo(t, dsn, 10)
	pool := openScratch(t, dsn)
	docID := uuid.New()
	insertReadyDocument(t, pool, docID)
	pool.Close()
	migrateUp(t, dsn)
	pool = openScratch(t, dsn)
	defer pool.Close()
	var status string
	if err := pool.QueryRow(context.Background(), `SELECT processing_status FROM documents WHERE id = $1`, docID).Scan(&status); err != nil {
		t.Fatal(err)
	}
	if status != "UPLOADED" {
		t.Fatalf("unlinked document status %s", status)
	}
	assertVersion(t, pool, 16)
	_ = name
}

func TestUpgradeFromV10FailsClosedWhenCommuneCannotBeDetermined(t *testing.T) {
	_, dsn := scratchDB(t)
	migrateTo(t, dsn, 10)
	pool := openScratch(t, dsn)
	_, err := pool.Exec(context.Background(), `
		INSERT INTO admin_write_idempotency (request_id, action, payload_hash, status_code, response_json)
		VALUES ($1, 'LINK', 'abc', 201, '{}'::jsonb)`, uuid.New())
	pool.Close()
	if err != nil {
		t.Fatal(err)
	}
	if err := migrateUpErr(dsn); err == nil {
		t.Fatal("migration accepted an idempotency row with no commune")
	}
}

func TestLegacyIdempotencyReplaysAfterUpgrade(t *testing.T) {
	_, dsn := scratchDB(t)
	migrateTo(t, dsn, 10)
	pool := openScratch(t, dsn)
	docID := uuid.New()
	insertReadyDocument(t, pool, docID)
	var versionID, procedureID uuid.UUID
	var code, version string
	if err := pool.QueryRow(context.Background(), `
		SELECT v.id, p.id, p.procedure_code, v.version
		FROM procedure_versions v
		JOIN procedures p ON p.id = v.procedure_id
		WHERE p.xa_id = 'xa_chu_se' AND p.domain_id = 'ho_tich_chung_thuc' AND v.status <> 'ARCHIVED'
		LIMIT 1`).Scan(&versionID, &procedureID, &code, &version); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(context.Background(), `
		INSERT INTO procedure_version_documents (
			procedure_version_id, document_id, procedure_id, xa_id, domain_id, relationship_type
		) VALUES ($1,$2,$3,'xa_chu_se','ho_tich_chung_thuc','SOURCE')`, versionID, docID, procedureID); err != nil {
		t.Fatal(err)
	}
	linkReq := uuid.New()
	unlinkReq := uuid.New()
	linkHash := legacyPayloadHash("LINK", docID, versionID, "SOURCE", "")
	unlinkHash := legacyPayloadHash("UNLINK", docID, versionID, "", "")
	linkBody, _ := json.Marshal(map[string]string{
		"document_id":          docID.String(),
		"procedure_id":         procedureID.String(),
		"procedure_version_id": versionID.String(),
		"procedure_code":       code,
		"version":              version,
		"relationship_type":    "SOURCE",
	})
	if _, err := pool.Exec(context.Background(), `
		INSERT INTO audit_logs (request_id, action, entity_type, entity_id, payload)
		VALUES ($1, 'DOCUMENT_LINKED', 'document', $2, jsonb_build_object('xa_id','xa_chu_se','procedure_version_id',$3::text))`,
		linkReq, docID.String(), versionID.String()); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(context.Background(), `
		INSERT INTO audit_logs (request_id, action, entity_type, entity_id, payload)
		VALUES ($1, 'DOCUMENT_UNLINKED', 'document', $2, jsonb_build_object('xa_id','xa_chu_se','procedure_version_id',$3::text))`,
		unlinkReq, docID.String(), versionID.String()); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(context.Background(), `
		INSERT INTO admin_write_idempotency (request_id, action, payload_hash, status_code, response_json)
		VALUES ($1, 'LINK', $2, 201, $3::jsonb)`, linkReq, linkHash, linkBody); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(context.Background(), `
		INSERT INTO admin_write_idempotency (request_id, action, payload_hash, status_code, response_json)
		VALUES ($1, 'UNLINK', $2, 200, '{"status":"unlinked"}'::jsonb)`, unlinkReq, unlinkHash); err != nil {
		t.Fatal(err)
	}
	pool.Close()
	migrateUp(t, dsn)
	pool = openScratch(t, dsn)
	defer pool.Close()
	eng, admin := engine(t, pool, readyWorker(), true)
	replay := call(t, eng, http.MethodPost, "/api/v1/admin/documents/"+docID.String()+"/links", admin, linkReq, map[string]string{
		"procedure_version_id": versionID.String(), "relationship_type": "SOURCE",
	})
	if replay.Code != http.StatusOK {
		t.Fatalf("link replay %d %s", replay.Code, replay.Body.String())
	}
	var env struct {
		Data struct {
			IndexStatus string    `json:"index_status"`
			UpdatedAt   time.Time `json:"updated_at"`
			Recoverable *bool     `json:"recoverable"`
		} `json:"data"`
	}
	if err := json.Unmarshal(replay.Body.Bytes(), &env); err != nil {
		t.Fatal(err)
	}
	if env.Data.IndexStatus == "" || env.Data.UpdatedAt.IsZero() || env.Data.Recoverable == nil {
		t.Fatalf("legacy link replay %+v", env.Data)
	}
	var links int
	if err := pool.QueryRow(context.Background(), `SELECT count(*) FROM procedure_version_documents WHERE document_id = $1`, docID).Scan(&links); err != nil {
		t.Fatal(err)
	}
	if links != 1 {
		t.Fatalf("link replay changed the row count to %d", links)
	}
	conflict := call(t, eng, http.MethodPost, "/api/v1/admin/documents/"+docID.String()+"/links", admin, linkReq, map[string]string{
		"procedure_version_id": versionID.String(), "relationship_type": "SOURCE", "page_range": "1-2",
	})
	if conflict.Code != http.StatusConflict {
		t.Fatalf("payload conflict %d %s", conflict.Code, conflict.Body.String())
	}
	unlinked := call(t, eng, http.MethodDelete, "/api/v1/admin/documents/"+docID.String()+"/links/"+versionID.String(), admin, unlinkReq, nil)
	if unlinked.Code != http.StatusOK {
		t.Fatalf("unlink replay %d %s", unlinked.Code, unlinked.Body.String())
	}
	if err := pool.QueryRow(context.Background(), `SELECT count(*) FROM procedure_version_documents WHERE document_id = $1`, docID).Scan(&links); err != nil {
		t.Fatal(err)
	}
	if links != 1 {
		t.Fatalf("unlink replay deleted the link, count %d", links)
	}
	other := otherCommuneDoc(t, pool)
	hidden := call(t, eng, http.MethodPost, "/api/v1/admin/documents/"+other+"/links", admin, linkReq, map[string]string{
		"procedure_version_id": versionID.String(), "relationship_type": "SOURCE",
	})
	if hidden.Code != http.StatusNotFound || strings.Contains(hidden.Body.String(), docID.String()) {
		t.Fatalf("cross commune %d %s", hidden.Code, hidden.Body.String())
	}
}

func legacyPayloadHash(kind string, documentID, versionID uuid.UUID, rel, page string) string {
	sum := sha256.Sum256([]byte(kind + "\n" + documentID.String() + "\n" + versionID.String() + "\n" + rel + "\n" + page))
	return hex.EncodeToString(sum[:])
}

func insertReadyDocument(t *testing.T, pool *pgxpool.Pool, docID uuid.UUID) {
	t.Helper()
	sum := legacyChecksum(docID)
	_, err := pool.Exec(context.Background(), `
		INSERT INTO documents (
			id, xa_id, domain_id, title, filename, storage_uri, checksum, mime_type, file_size_bytes,
			processing_status, validity_status, uploaded_by
		) VALUES (
			$1,'xa_chu_se','ho_tich_chung_thuc','Ready khong link','a.pdf','s3://cas-documents/x',
			$2,'application/pdf', 8, 'READY', 'PENDING', 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'
		)`, docID, sum)
	if err != nil {
		t.Fatal(err)
	}
}

func legacyChecksum(id uuid.UUID) string {
	hexID := ""
	for _, c := range id.String() {
		if c != '-' {
			hexID += string(c)
		}
	}
	return hexID + hexID
}

var scratchName = regexp.MustCompile(`^[a-z][a-z0-9_]{0,40}$`)

func scratchDB(t *testing.T) (string, string) {
	t.Helper()
	if os.Getenv("CAS_INTEGRATION") != "1" {
		t.Fatal("CAS_INTEGRATION=1 required")
	}
	name := "cas_p4a_" + uuid.NewString()[:8]
	if !scratchName.MatchString(name) {
		t.Fatalf("bad scratch name %s", name)
	}
	ctx := context.Background()
	admin, err := pgxpool.New(ctx, maintenanceDSN(t))
	if err != nil {
		t.Fatal(err)
	}
	defer admin.Close()
	if _, err := admin.Exec(ctx, "CREATE DATABASE "+name); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { dropScratch(name) })
	return name, databaseDSN(t, name)
}

func dropScratch(name string) {
	if !scratchName.MatchString(name) {
		return
	}
	ctx := context.Background()
	admin, err := pgxpool.New(ctx, maintenanceDSN(nil))
	if err != nil {
		return
	}
	defer admin.Close()
	_, _ = admin.Exec(ctx, `SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = $1 AND pid <> pg_backend_pid()`, name)
	_, _ = admin.Exec(ctx, "DROP DATABASE IF EXISTS "+name)
}

func maintenanceDSN(t *testing.T) string {
	return databaseDSN(t, "postgres")
}

func databaseDSN(t *testing.T, name string) string {
	raw := os.Getenv("DATABASE_URL")
	if raw == "" {
		raw = "postgres://cas:cas@127.0.0.1:5432/citizen_assistance?sslmode=disable"
	}
	u, err := url.Parse(raw)
	if err != nil {
		if t != nil {
			t.Fatal(err)
		}
		return raw
	}
	u.Path = "/" + name
	return u.String()
}

func migrateTo(t *testing.T, dsn string, version uint) {
	t.Helper()
	m := newMigrator(t, dsn)
	defer m.Close()
	if err := m.Migrate(version); err != nil && err != migrate.ErrNoChange {
		t.Fatal(err)
	}
}

func migrateUp(t *testing.T, dsn string) {
	t.Helper()
	if err := migrateUpErr(dsn); err != nil {
		t.Fatal(err)
	}
}

func migrateUpErr(dsn string) error {
	m, err := migrate.New(migrationSource(), dsn)
	if err != nil {
		return err
	}
	defer m.Close()
	err = m.Up()
	if err == migrate.ErrNoChange {
		return nil
	}
	return err
}

func newMigrator(t *testing.T, dsn string) *migrate.Migrate {
	t.Helper()
	m, err := migrate.New(migrationSource(), dsn)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func migrationSource() string {
	root, err := filepath.Abs(filepath.Join("..", "..", "..", ".."))
	if err != nil {
		return ""
	}
	u := url.URL{Scheme: "file", Path: filepath.ToSlash(filepath.Join(root, "deploy", "migrations"))}
	return u.String()
}

func openScratch(t *testing.T, dsn string) *pgxpool.Pool {
	t.Helper()
	pool, err := pgxpool.New(context.Background(), dsn)
	if err != nil {
		t.Fatal(err)
	}
	return pool
}

func TestMigration16RoundTripKeepsGenerationRows(t *testing.T) {
	_, dsn := scratchDB(t)
	migrateTo(t, dsn, 15)
	pool := openScratch(t, dsn)
	var before int
	if err := pool.QueryRow(context.Background(), `SELECT count(*) FROM document_index_generations`).Scan(&before); err != nil {
		t.Fatal(err)
	}
	pool.Close()
	migrateUp(t, dsn)
	pool = openScratch(t, dsn)
	assertVersion(t, pool, 16)
	var dirty bool
	if err := pool.QueryRow(context.Background(), `SELECT dirty FROM schema_migrations`).Scan(&dirty); err != nil {
		t.Fatal(err)
	}
	if dirty {
		t.Fatal("dirty after up")
	}
	var after int
	if err := pool.QueryRow(context.Background(), `SELECT count(*) FROM document_index_generations`).Scan(&after); err != nil {
		t.Fatal(err)
	}
	if after != before {
		t.Fatalf("generations %d -> %d", before, after)
	}
	var status string
	if err := pool.QueryRow(context.Background(), `
		SELECT column_default FROM information_schema.columns
		WHERE table_name = 'document_index_generations' AND column_name = 'cleanup_status'`).Scan(&status); err != nil {
		t.Fatal(err)
	}
	pool.Close()
	m := newMigrator(t, dsn)
	if err := m.Steps(-1); err != nil {
		t.Fatal(err)
	}
	m.Close()
	pool = openScratch(t, dsn)
	assertVersion(t, pool, 15)
	pool.Close()
	migrateUp(t, dsn)
	pool = openScratch(t, dsn)
	defer pool.Close()
	assertVersion(t, pool, 16)
	if err := pool.QueryRow(context.Background(), `SELECT dirty FROM schema_migrations`).Scan(&dirty); err != nil {
		t.Fatal(err)
	}
	if dirty {
		t.Fatal("dirty after down/up")
	}
}

func assertVersion(t *testing.T, pool *pgxpool.Pool, want int) {
	t.Helper()
	var version int
	if err := pool.QueryRow(context.Background(), `SELECT version FROM schema_migrations`).Scan(&version); err != nil {
		t.Fatal(err)
	}
	if version != want {
		t.Fatalf("version %d", version)
	}
}
