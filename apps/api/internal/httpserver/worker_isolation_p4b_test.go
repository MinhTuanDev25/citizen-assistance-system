package httpserver_test

import (
	"strings"
	"testing"
)

func TestP4BWorkerIdentityRejectsSharedPostgres(t *testing.T) {
	id := expectedP4BWorkerIdentity()
	id.PostgresPort = "5432"
	id.Database = "citizen_assistance"
	if err := id.Validate(); err == nil {
		t.Fatal("shared postgres must be rejected")
	}
}

func TestP4BWorkerIdentityRejectsSharedMinIO(t *testing.T) {
	id := expectedP4BWorkerIdentity()
	id.MinioEndpoint = "127.0.0.1:9000"
	id.MinioBucket = "cas-documents"
	if err := id.Validate(); err == nil {
		t.Fatal("shared minio must be rejected")
	}
}

func TestP4BWorkerIdentityRejectsSharedQdrant(t *testing.T) {
	id := expectedP4BWorkerIdentity()
	id.QdrantURL = "http://127.0.0.1:6333"
	if err := id.Validate(); err == nil {
		t.Fatal("shared qdrant must be rejected")
	}
}

func TestP4BWorkerIdentityRejectsEmpty(t *testing.T) {
	if err := (p4bWorkerIdentity{}).Validate(); err == nil {
		t.Fatal("empty identity must be rejected")
	}
}

func TestP4BWorkerIdentityAcceptsP4B(t *testing.T) {
	if err := expectedP4BWorkerIdentity().Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestP4BWorkerIdentityReadyWithoutInspectIsRejected(t *testing.T) {
	err := verifyP4BWorker("http://127.0.0.1:9", p4bWorkerIdentity{})
	if err == nil {
		t.Fatal("uninspected worker must be rejected")
	}
}

func TestP4BWorkerIdentityUnavailableEndpoint(t *testing.T) {
	err := verifyP4BWorker("http://127.0.0.1:1", expectedP4BWorkerIdentity())
	if err == nil || !strings.Contains(err.Error(), "unavailable") && !strings.Contains(err.Error(), "worker") {
		t.Fatalf("got %v", err)
	}
}

func TestP4BRejectsDeclaredWorkerURLReuse(t *testing.T) {
	if err := rejectDeclaredP4BWorkerURL("http://127.0.0.1:8011"); err == nil {
		t.Fatal("declared worker URL must be rejected")
	}
	if err := rejectDeclaredP4BWorkerURL(""); err != nil {
		t.Fatal(err)
	}
}

func TestP4BCleanupDoesNotTouchUnrelatedW2E(t *testing.T) {
	pool := p4bWorkerPool(t)
	ctx, cancel := contextWithTimeout()
	defer cancel()
	admin := mustAdmin()
	unrelated := "w2e_unrelated_" + shortID()
	var procID, verID string
	if err := pool.QueryRow(ctx, `
		INSERT INTO procedures (procedure_code, domain_id, name, xa_id)
		VALUES ($1,'ho_tich_chung_thuc','unrelated sentinel','xa_chu_se')
		RETURNING id::text`, unrelated).Scan(&procID); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(ctx, `
		INSERT INTO procedure_versions (procedure_id, version, status, definition, created_by, approved_by, approved_at)
		VALUES ($1::uuid,'9.9.9','ACTIVE','{"procedure_code":"x","name":"x","intent_examples":["x"],"slots":{},"required_slots":[],"guidance":{"summary":"x","checklist":[]}}'::jsonb,$2,$2,now())
		RETURNING id::text`, procID, admin).Scan(&verID); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `UPDATE procedures SET active_version_id=$1::uuid WHERE id=$2::uuid`, verID, procID); err != nil {
		t.Fatal(err)
	}
	fixtureProc := "w2e_fixture_" + shortID()
	var fxID string
	if err := pool.QueryRow(ctx, `
		INSERT INTO procedures (procedure_code, domain_id, name, xa_id)
		VALUES ($1,'ho_tich_chung_thuc','fixture','xa_chu_se') RETURNING id::text`, fixtureProc).Scan(&fxID); err != nil {
		t.Fatal(err)
	}
	archiveFixtureOnly(ctx, pool, fxID)
	var status string
	if err := pool.QueryRow(ctx, `SELECT v.status FROM procedure_versions v WHERE v.id=$1::uuid`, verID).Scan(&status); err != nil {
		t.Fatal(err)
	}
	if status != "ACTIVE" {
		t.Fatalf("unrelated w2e_* was modified: status=%s", status)
	}
}
