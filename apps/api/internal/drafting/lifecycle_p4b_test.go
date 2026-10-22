package drafting

import (
	"context"
	"encoding/json"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/activation"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/repository"
)

const p4bDSN = "postgres://cas:cas@127.0.0.1:5434/citizen_assistance_p4b_test?sslmode=disable"

func p4bPool(t *testing.T) *pgxpool.Pool {
	t.Helper()
	cfg, err := pgxpool.ParseConfig(p4bDSN)
	if err != nil {
		t.Fatal(err)
	}
	if parsed, err := url.Parse(p4bDSN); err == nil {
		port, _ := strconv.Atoi(parsed.Port())
		if parsed.Hostname() != "127.0.0.1" || port != 5434 || strings.Trim(parsed.Path, "/") != "citizen_assistance_p4b_test" {
			t.Fatal("p4b DSN drifted")
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	pool, err := pgxpool.NewWithConfig(ctx, cfg)
	if err != nil {
		t.Skip("p4b-test postgres not reachable")
	}
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		t.Skip("p4b-test postgres not reachable")
	}
	var name string
	if err := pool.QueryRow(ctx, `SELECT current_database()`).Scan(&name); err != nil || name != "citizen_assistance_p4b_test" {
		pool.Close()
		t.Skip("not citizen_assistance_p4b_test")
	}
	t.Cleanup(pool.Close)
	return pool
}

func TestP4BActivationResumeReconcileAndPin(t *testing.T) {
	pool := p4bPool(t)
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	admin := uuid.MustParse("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
	code := "w2c_" + strings.ReplaceAll(uuid.NewString(), "-", "")[:12]
	var procID uuid.UUID
	if err := pool.QueryRow(ctx, `
		INSERT INTO procedures (procedure_code, domain_id, name, xa_id)
		VALUES ($1, 'ho_tich_chung_thuc', 'Wave 2 closure fixture', 'xa_chu_se')
		ON CONFLICT (xa_id, procedure_code) DO UPDATE SET name = EXCLUDED.name
		RETURNING id`, code).Scan(&procID); err != nil {
		t.Fatal(err)
	}
	docID := uuid.New()
	if _, err := pool.Exec(ctx, `
		INSERT INTO documents (
			id, xa_id, domain_id, title, filename, storage_uri, checksum, mime_type, file_size_bytes,
			processing_status, validity_status, uploaded_by
		) VALUES (
			$1,'xa_chu_se','ho_tich_chung_thuc','closure source','c.pdf','s3://cas-p4b-integration-test/p4b/test.pdf',$2,
			'application/pdf', 12, 'READY', 'VALID', $3
		)`, docID, strings.ReplaceAll(docID.String(), "-", "")+strings.ReplaceAll(docID.String(), "-", ""), admin); err != nil {
		t.Fatal(err)
	}

	def := map[string]any{
		"procedure_code":  code,
		"name":            "Wave 2 closure fixture",
		"intent_examples": []string{"fixture closure"},
		"slots":           map[string]any{"noi_sinh": map[string]any{"type": "string", "question": "Nơi sinh?"}},
		"required_slots":  []string{"noi_sinh"},
		"guidance":        map[string]any{"summary": "Nộp tại UBND cấp xã", "checklist": []string{}},
		"citations":       []map[string]any{{"doc_id": docID.String(), "title": "fixture", "source_type": "ai_draft_unapproved"}},
	}
	raw, _ := json.Marshal(def)
	ex := testExcerpt()
	ids := []string{"slot:noi_sinh", "required:noi_sinh", "guidance:summary"}
	items := make([]EvidenceItem, 0, len(ids))
	for _, id := range ids {
		item := validItem(id, "UBND cấp xã")
		item.ReviewStatus = ReviewAccepted
		items = append(items, item)
	}
	view := DraftView{
		ProcedureID: procID, ProcedureCode: code, DocumentID: docID,
		Definition: raw, Evidence: items, Status: repository.DraftStatusApproved,
		OriginalProcedureCode: code, SourcePageStart: 10, SourcePageEnd: 12,
	}
	revalidate(&view, SourceSnapshot{Excerpts: []Excerpt{ex}}, code, code)
	if err := applyReviewDecisions(&view, []ReviewDecision{
		{ProposalID: "slot:noi_sinh", Decision: ReviewAccepted},
		{ProposalID: "required:noi_sinh", Decision: ReviewAccepted},
		{ProposalID: "guidance:summary", Decision: ReviewAccepted},
	}); err != nil {
		t.Fatal(err)
	}
	if err := approvalGate(view); err != nil {
		t.Fatalf("fixture must be approvable: %v issues=%v", err, view.Validation.Issues)
	}
	view.Validation.SemanticApproved = true
	payload, _ := json.Marshal(storedPayloadFromView(view))
	validationRaw, _ := json.Marshal(view.Validation)
	draftID := uuid.New()
	now := time.Now().UTC()
	if _, err := pool.Exec(ctx, `
		INSERT INTO procedure_drafts (
			id, document_id, procedure_id, draft_definition, validation_result, status, created_by, updated_by
		) VALUES ($1,$2,$3,$4,$5,'APPROVED',$6,$6)`,
		draftID, docID, procID, payload, validationRaw, admin); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(ctx, `SELECT updated_at FROM procedure_drafts WHERE id=$1`, draftID).Scan(&now); err != nil {
		t.Fatal(err)
	}

	drafts := &repository.DraftRepo{Pool: pool}
	svc := &Service{
		XAID:       "xa_chu_se",
		Source:     &SourceStore{Pool: pool},
		Drafts:     drafts,
		Activation: &activation.Service{Store: &activation.Store{Pool: pool}, XAID: "xa_chu_se"},
	}
	first, err := svc.Activate(ctx, admin, uuid.New(), draftID, ActivateDraftRequest{Confirm: true, ExpectedUpdatedAt: &now})
	if err != nil {
		t.Fatalf("create version: %v", err)
	}
	if first.Activated || first.Version == nil {
		t.Fatalf("expected version without activation, got %+v", first)
	}
	versionID := first.Version.ID

	if err := markSourceReady(ctx, pool, procID, versionID, docID, admin); err != nil {
		t.Fatal(err)
	}

	row, err := svc.Drafts.Get(ctx, "xa_chu_se", draftID)
	if err != nil {
		t.Fatal(err)
	}
	var currentActive *uuid.UUID
	if err := pool.QueryRow(ctx, `SELECT active_version_id FROM procedures WHERE id=$1`, procID).Scan(&currentActive); err != nil {
		t.Fatal(err)
	}
	svc.Drafts = &failUpdateStatus{DraftRepo: drafts, on: repository.DraftStatusPublished, err: errPublishInjected}
	second, err := svc.Activate(ctx, admin, uuid.New(), draftID, ActivateDraftRequest{
		Confirm: true, ExpectedUpdatedAt: &row.UpdatedAt, ExpectedActiveVersionID: currentActive,
	})
	if err != nil {
		t.Fatalf("activate with hook: %v", err)
	}
	if !second.Activated || !second.ReconciliationRequired {
		t.Fatalf("expected ACTIVE + reconcile pending, got %+v", second)
	}
	if second.Draft.Status != repository.DraftStatusApproved {
		t.Fatalf("draft status=%s", second.Draft.Status)
	}
	var verStatus string
	if err := pool.QueryRow(ctx, `SELECT status FROM procedure_versions WHERE id=$1`, versionID).Scan(&verStatus); err != nil {
		t.Fatal(err)
	}
	if verStatus != "ACTIVE" {
		t.Fatalf("version status=%s", verStatus)
	}

	svc.Drafts = drafts
	third, err := svc.Activate(ctx, admin, uuid.New(), draftID, ActivateDraftRequest{Confirm: true})
	if err != nil {
		t.Fatalf("reconcile: %v", err)
	}
	if !third.Activated || third.Draft.Status != repository.DraftStatusPublished {
		t.Fatalf("reconcile result %+v status=%s", third, third.Draft.Status)
	}

	prev := versionID
	sessionID := uuid.New()
	if _, err := pool.Exec(ctx, `
		INSERT INTO conversation_sessions (id, xa_id, guest_token, active_procedure_id, active_procedure_version_id, status)
		VALUES ($1,'xa_chu_se',$2,$3,$4,'OPEN')`, sessionID, "pin-"+uuid.NewString(), procID, prev); err != nil {
		t.Fatal(err)
	}
	var pinned uuid.UUID
	if err := pool.QueryRow(ctx, `SELECT active_procedure_version_id FROM conversation_sessions WHERE id=$1`, sessionID).Scan(&pinned); err != nil {
		t.Fatal(err)
	}
	if pinned != prev {
		t.Fatalf("pin drifted %s", pinned)
	}

	fourth, err := svc.Activate(ctx, admin, uuid.New(), draftID, ActivateDraftRequest{Confirm: true})
	if err != nil {
		t.Fatal(err)
	}
	if !fourth.Activated || fourth.Version == nil || fourth.Version.ID != versionID {
		t.Fatalf("retry must not create another version: %+v", fourth)
	}
}

func TestP4BPublishStatusUpdateFailureKeepsApproved(t *testing.T) {
	pool := p4bPool(t)
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	fx := seedApprovedDraft(t, ctx, pool)
	svc := fx.service()
	first, err := svc.Activate(ctx, fx.admin, uuid.New(), fx.draftID, ActivateDraftRequest{Confirm: true, ExpectedUpdatedAt: &fx.updatedAt})
	if err != nil {
		t.Fatal(err)
	}
	if err := markSourceReady(ctx, pool, fx.procID, first.Version.ID, fx.docID, fx.admin); err != nil {
		t.Fatal(err)
	}
	row, err := svc.Drafts.Get(ctx, "xa_chu_se", fx.draftID)
	if err != nil {
		t.Fatal(err)
	}
	repo, _ := svc.Drafts.(*repository.DraftRepo)
	svc.Drafts = &failUpdateStatus{DraftRepo: repo, on: repository.DraftStatusPublished, err: errString("injected UpdateStatus failure")}
	second, err := svc.Activate(ctx, fx.admin, uuid.New(), fx.draftID, ActivateDraftRequest{Confirm: true, ExpectedUpdatedAt: &row.UpdatedAt})
	if err != nil {
		t.Fatalf("activate: %v", err)
	}
	if !second.Activated || !second.ReconciliationRequired {
		t.Fatalf("expected reconcile pending, got %+v", second)
	}
	if second.Draft.Status != repository.DraftStatusApproved {
		t.Fatalf("response status=%s", second.Draft.Status)
	}
	var dbStatus string
	if err := pool.QueryRow(ctx, `SELECT status FROM procedure_drafts WHERE id=$1`, fx.draftID).Scan(&dbStatus); err != nil {
		t.Fatal(err)
	}
	if dbStatus != repository.DraftStatusApproved {
		t.Fatalf("db status=%s", dbStatus)
	}
}

func TestP4BConcurrentActivateOneVersion(t *testing.T) {
	pool := p4bPool(t)
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	fx := seedApprovedDraft(t, ctx, pool)
	svc := fx.service()
	var wg sync.WaitGroup
	type outcome struct {
		res ActivateDraftResult
		err error
	}
	out := make([]outcome, 2)
	wg.Add(2)
	for i := 0; i < 2; i++ {
		i := i
		go func() {
			defer wg.Done()
			res, err := svc.Activate(ctx, fx.admin, uuid.New(), fx.draftID, ActivateDraftRequest{Confirm: true, ExpectedUpdatedAt: &fx.updatedAt})
			out[i] = outcome{res, err}
		}()
	}
	wg.Wait()
	var versionIDs []uuid.UUID
	for i, item := range out {
		if item.err != nil {
			t.Fatalf("caller %d: %v", i, item.err)
		}
		if item.res.Version == nil {
			t.Fatalf("caller %d missing version %+v", i, item.res)
		}
		versionIDs = append(versionIDs, item.res.Version.ID)
	}
	if versionIDs[0] != versionIDs[1] {
		t.Fatalf("duplicate versions %s vs %s", versionIDs[0], versionIDs[1])
	}
	var n int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM procedure_versions WHERE source_draft_id=$1`, fx.draftID).Scan(&n); err != nil {
		t.Fatal(err)
	}
	if n != 1 {
		t.Fatalf("versions for draft=%d", n)
	}
}

func TestP4BIndexMetadataPersistFailureIsReported(t *testing.T) {
	view := DraftView{Status: repository.DraftStatusApproved, Definition: json.RawMessage(`{"procedure_code":"x","name":"x"}`)}
	svc := &Service{Drafts: &failUpdateStatus{on: repository.DraftStatusApproved, err: errString("injected persist failure")}}
	_, err := svc.persistApprovedPayload(context.Background(), uuid.New(), uuid.New(), time.Time{}, view)
	if err == nil {
		t.Fatal("expected persist failure")
	}
}

type approvedFixture struct {
	admin     uuid.UUID
	procID    uuid.UUID
	docID     uuid.UUID
	draftID   uuid.UUID
	code      string
	updatedAt time.Time
	pool      *pgxpool.Pool
}

func (fx approvedFixture) service() *Service {
	return &Service{
		XAID:       "xa_chu_se",
		Source:     &SourceStore{Pool: fx.pool},
		Drafts:     &repository.DraftRepo{Pool: fx.pool},
		Activation: &activation.Service{Store: &activation.Store{Pool: fx.pool}, XAID: "xa_chu_se"},
	}
}

func seedApprovedDraft(t *testing.T, ctx context.Context, pool *pgxpool.Pool) approvedFixture {
	t.Helper()
	admin := uuid.MustParse("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
	code := "w2c_" + strings.ReplaceAll(uuid.NewString(), "-", "")[:12]
	var procID uuid.UUID
	if err := pool.QueryRow(ctx, `
		INSERT INTO procedures (procedure_code, domain_id, name, xa_id)
		VALUES ($1, 'ho_tich_chung_thuc', 'Wave 2 closure fixture', 'xa_chu_se')
		ON CONFLICT (xa_id, procedure_code) DO UPDATE SET name = EXCLUDED.name
		RETURNING id`, code).Scan(&procID); err != nil {
		t.Fatal(err)
	}
	docID := uuid.New()
	if _, err := pool.Exec(ctx, `
		INSERT INTO documents (
			id, xa_id, domain_id, title, filename, storage_uri, checksum, mime_type, file_size_bytes,
			processing_status, validity_status, uploaded_by
		) VALUES (
			$1,'xa_chu_se','ho_tich_chung_thuc','closure source','c.pdf','s3://cas-p4b-integration-test/p4b/test.pdf',$2,
			'application/pdf', 12, 'READY', 'VALID', $3
		)`, docID, strings.ReplaceAll(docID.String(), "-", "")+strings.ReplaceAll(docID.String(), "-", ""), admin); err != nil {
		t.Fatal(err)
	}
	def := map[string]any{
		"procedure_code":  code,
		"name":            "Wave 2 closure fixture",
		"intent_examples": []string{"fixture closure"},
		"slots":           map[string]any{"noi_sinh": map[string]any{"type": "string", "question": "Nơi sinh?"}},
		"required_slots":  []string{"noi_sinh"},
		"guidance":        map[string]any{"summary": "Nộp tại UBND cấp xã", "checklist": []string{}},
		"citations":       []map[string]any{{"doc_id": docID.String(), "title": "fixture", "source_type": "ai_draft_unapproved"}},
	}
	raw, _ := json.Marshal(def)
	ex := testExcerpt()
	ids := []string{"slot:noi_sinh", "required:noi_sinh", "guidance:summary"}
	items := make([]EvidenceItem, 0, len(ids))
	for _, id := range ids {
		item := validItem(id, "UBND cấp xã")
		item.ReviewStatus = ReviewAccepted
		items = append(items, item)
	}
	view := DraftView{
		ProcedureID: procID, ProcedureCode: code, DocumentID: docID,
		Definition: raw, Evidence: items, Status: repository.DraftStatusApproved,
		OriginalProcedureCode: code, SourcePageStart: 10, SourcePageEnd: 12,
	}
	revalidate(&view, SourceSnapshot{Excerpts: []Excerpt{ex}}, code, code)
	if err := applyReviewDecisions(&view, []ReviewDecision{
		{ProposalID: "slot:noi_sinh", Decision: ReviewAccepted},
		{ProposalID: "required:noi_sinh", Decision: ReviewAccepted},
		{ProposalID: "guidance:summary", Decision: ReviewAccepted},
	}); err != nil {
		t.Fatal(err)
	}
	view.Validation.SemanticApproved = true
	payload, _ := json.Marshal(storedPayloadFromView(view))
	validationRaw, _ := json.Marshal(view.Validation)
	draftID := uuid.New()
	if _, err := pool.Exec(ctx, `
		INSERT INTO procedure_drafts (
			id, document_id, procedure_id, draft_definition, validation_result, status, created_by, updated_by
		) VALUES ($1,$2,$3,$4,$5,'APPROVED',$6,$6)`,
		draftID, docID, procID, payload, validationRaw, admin); err != nil {
		t.Fatal(err)
	}
	var updated time.Time
	if err := pool.QueryRow(ctx, `SELECT updated_at FROM procedure_drafts WHERE id=$1`, draftID).Scan(&updated); err != nil {
		t.Fatal(err)
	}
	return approvedFixture{admin: admin, procID: procID, docID: docID, draftID: draftID, code: code, updatedAt: updated, pool: pool}
}

var errPublishInjected = errString("injected publish failure")

type errString string

func (e errString) Error() string { return string(e) }

func markSourceReady(ctx context.Context, pool *pgxpool.Pool, procID, versionID, docID, admin uuid.UUID) error {
	jobID := uuid.New()
	genID := uuid.New()
	sum := strings.ReplaceAll(docID.String(), "-", "")
	sum = sum + sum
	if _, err := pool.Exec(ctx, `
		INSERT INTO procedure_version_documents (
			procedure_version_id, document_id, procedure_id, xa_id, domain_id, relationship_type, index_status
		) VALUES ($1,$2,$3,'xa_chu_se','ho_tich_chung_thuc','SOURCE','UPLOADED')`, versionID, docID, procID); err != nil {
		if !strings.Contains(err.Error(), "duplicate") && !strings.Contains(err.Error(), "unique") {
			return err
		}
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO document_index_jobs (
			id, document_id, xa_id, request_id, payload_hash, status, procedure_version_id, claim_token, finished_at
		) VALUES ($1,$2,'xa_chu_se',$3,'hash','SUCCEEDED',$4,gen_random_uuid(), now())`,
		jobID, docID, uuid.New(), versionID); err != nil {
		return err
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
		)`, genID, docID, versionID, jobID, sum); err != nil {
		return err
	}
	_, err := pool.Exec(ctx, `
		UPDATE procedure_version_documents
		SET index_status = 'READY', active_generation_id = $3
		WHERE procedure_version_id = $1 AND document_id = $2`, versionID, docID, genID)
	return err
}
