package drafting

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgconn"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/decision"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/repository"
)

// ---- synthetic multi-procedure document ------------------------------------

var (
	ldDoc  = uuid.MustParse("d0000000-0000-0000-0000-000000000001")
	ldGen  = uuid.MustParse("e0000000-0000-0000-0000-000000000001")
	ldProc = uuid.MustParse("f0000000-0000-0000-0000-000000000001")
	ldUser = uuid.MustParse("a0000000-0000-0000-0000-000000000001")
)

const (
	ldHeading1   = "A.THỦ TỤC HÀNH CHÍNH CẤP XÃ 1.Thủ tục đăng ký khai sinh Trình tự thực hiện: người yêu cầu nộp hồ sơ tại Ủy ban nhân dân cấp xã."
	ldHinhThuc   = "Người yêu cầu lựa chọn hình thức nộp hồ sơ trực tiếp hoặc trực tuyến."
	ldThanhPhan  = "Thành phần, số lượng hồ sơ: Tờ khai đăng ký khai sinh theo mẫu."
	ldMienPhi    = "Lệ phí: Miễn lệ phí đăng ký khai sinh đúng hạn."
	ldCachThuc   = "Cách thức thực hiện: nộp trực tiếp tại Trung tâm Phục vụ hành chính công."
	ldBuuChinh   = "Người yêu cầu có thể gửi hồ sơ qua hệ thống bưu chính."
	ldHoiGCS     = "Người yêu cầu nộp Giấy chứng sinh do cơ sở y tế cấp."
	ldBatBuocGCS = "Giấy chứng sinh là giấy tờ bắt buộc phải nộp khi đăng ký khai sinh."
	ldToKhai2    = "Tờ khai đăng ký khai sinh theo mẫu; Giấy chứng sinh."
	ldPhi8k      = "Lệ phí đăng ký khai sinh không đúng hạn là 8.000 đồng."
	ldThoiHan    = "Thời hạn giải quyết: trong ngày làm việc."
	ldHeading2   = "2.Thủ tục đăng ký lại khai sinh Trình tự thực hiện: nộp hồ sơ đăng ký lại khai sinh."
	ldHeading3   = "3.Thủ tục đăng ký kết hôn Trình tự thực hiện: hai bên nam nữ nộp tờ khai."
)

// Section 1 (đăng ký khai sinh) is chunks 2..48: 47 chunks = 2 batches (40 + 7).
func ldContent(i int) string {
	switch i {
	case 0:
		return "PHẦN I. DANH MỤC 1.001193 Đăng ký khai sinh 1.004884 Đăng ký lại khai sinh"
	case 1:
		return "PHẦN II. NỘI DUNG CỤ THỂ CỦA TỪNG THỦ TỤC HÀNH CHÍNH"
	case 2:
		return ldHeading1
	case 3:
		return ldHinhThuc
	case 4:
		return ldThanhPhan
	case 5:
		return ldMienPhi
	case 6:
		return ldCachThuc
	case 43:
		return ldBuuChinh
	case 44:
		return ldHoiGCS
	case 45:
		return ldBatBuocGCS
	case 46:
		return ldToKhai2
	case 47:
		return ldPhi8k
	case 48:
		return ldThoiHan
	case 49:
		return ldHeading2
	case 56:
		return ldHeading3
	}
	return fmt.Sprintf("Đoạn văn bản số %d của quyết định.", i)
}

func ldExcerpts() []Excerpt {
	out := make([]Excerpt, 0, 61)
	for i := 0; i <= 60; i++ {
		content := ldContent(i)
		sum := sha256.Sum256([]byte(content))
		page := 1 + i/5
		out = append(out, Excerpt{
			ExcerptID: uuid.NewSHA1(ldGen, []byte(fmt.Sprintf("chunk-%d", i))), DocumentID: ldDoc, GenerationID: ldGen,
			PageStart: page, PageEnd: page, TextSHA256: hex.EncodeToString(sum[:]), Content: content, ChunkIndex: i,
		})
	}
	return out
}

func ldChunk(i int) Excerpt { return ldExcerpts()[i] }

// ---- fakes ------------------------------------------------------------------

type fakeSource struct{ all []Excerpt }

func (f *fakeSource) snap() SourceSnapshot {
	return SourceSnapshot{
		ProcedureID: ldProc, ProcedureCode: "khai_sinh_v1", ProcedureName: "Đăng ký khai sinh", XaID: "xa_test",
		DocumentID: ldDoc, GenerationID: ldGen, GenerationStatus: "READY", GenerationPVStatus: "SOURCE",
	}
}

func (f *fakeSource) Resolve(_ context.Context, _ string, _, _, _ uuid.UUID, start, end int) (SourceSnapshot, error) {
	s := f.snap()
	for _, ex := range f.all {
		if ex.PageStart >= start && ex.PageEnd <= end {
			s.Excerpts = append(s.Excerpts, ex)
		}
	}
	s.SourcePageStart, s.SourcePageEnd = start, end
	return s, nil
}

func (f *fakeSource) ResolveChunks(_ context.Context, _ string, _, _, gen uuid.UUID, ranges []ChunkRange) (SourceSnapshot, error) {
	if gen != ldGen {
		return SourceSnapshot{}, ErrSourceNotFound
	}
	s := f.snap()
	for _, ex := range f.all {
		if len(ranges) == 0 || inRanges(ex.ChunkIndex, ranges) {
			s.Excerpts = append(s.Excerpts, ex)
		}
	}
	s.SourcePageStart, s.SourcePageEnd = excerptPageBounds(s.Excerpts)
	return s, nil
}

func (f *fakeSource) ListReady(context.Context, string, uuid.UUID) ([]SourceOption, error) {
	return []SourceOption{{DocumentID: ldDoc, GenerationID: ldGen}}, nil
}

type fakeDrafts struct {
	mu   sync.Mutex
	rows map[uuid.UUID]repository.ProcedureDraft
}

func newFakeDrafts() *fakeDrafts { return &fakeDrafts{rows: map[uuid.UUID]repository.ProcedureDraft{}} }

func (f *fakeDrafts) Insert(_ context.Context, row repository.ProcedureDraft) (repository.ProcedureDraft, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if row.ID == uuid.Nil {
		row.ID = uuid.New()
	}
	if _, ok := f.rows[row.ID]; ok {
		return repository.ProcedureDraft{}, &pgconn.PgError{Code: "23505"}
	}
	row.CreatedAt, row.UpdatedAt, row.ProcedureCode, row.XaID = time.Now(), time.Now(), "khai_sinh_v1", "xa_test"
	f.rows[row.ID] = row
	return row, nil
}

func (f *fakeDrafts) Get(_ context.Context, _ string, id uuid.UUID) (repository.ProcedureDraft, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	row, ok := f.rows[id]
	if !ok {
		return repository.ProcedureDraft{}, repository.ErrNotFound
	}
	return row, nil
}

func (f *fakeDrafts) ListByProcedure(_ context.Context, _ string, procedureID uuid.UUID) ([]repository.ProcedureDraft, error) {
	out := []repository.ProcedureDraft{}
	for _, row := range f.rows {
		if row.ProcedureID == procedureID {
			out = append(out, row)
		}
	}
	return out, nil
}

func (f *fakeDrafts) ListAll(context.Context, string, *uuid.UUID) ([]repository.ProcedureDraft, error) {
	return nil, nil
}

func (f *fakeDrafts) UpdateDefinition(_ context.Context, _ string, id, actor uuid.UUID, def, val json.RawMessage, expected time.Time) (repository.ProcedureDraft, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	row, ok := f.rows[id]
	if !ok || !row.UpdatedAt.Equal(expected) {
		return repository.ProcedureDraft{}, repository.ErrNotFound
	}
	row.DraftDefinition, row.ValidationResult, row.UpdatedBy = def, val, &actor
	row.UpdatedAt = row.UpdatedAt.Add(time.Second)
	f.rows[id] = row
	return row, nil
}

func (f *fakeDrafts) UpdateStatus(context.Context, string, uuid.UUID, uuid.UUID, string, json.RawMessage, json.RawMessage, time.Time, []string) (repository.ProcedureDraft, error) {
	return repository.ProcedureDraft{}, repository.ErrNotFound
}

// fakeJobs mimics repository.DraftJobRepo state transitions in memory.
type fakeJobs struct {
	mu      sync.Mutex
	jobs    map[uuid.UUID]*repository.DraftJob
	batches map[uuid.UUID][]repository.DraftJobBatch
	owner   map[uuid.UUID]uuid.UUID
}

func newFakeJobs() *fakeJobs {
	return &fakeJobs{jobs: map[uuid.UUID]*repository.DraftJob{}, batches: map[uuid.UUID][]repository.DraftJobBatch{}, owner: map[uuid.UUID]uuid.UUID{}}
}

func (f *fakeJobs) Create(_ context.Context, job repository.DraftJob, batches []repository.DraftJobBatch) (repository.DraftJob, bool, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	for _, j := range f.jobs {
		if j.RequestID == job.RequestID {
			if j.RequestHash != job.RequestHash {
				return repository.DraftJob{}, false, repository.ErrIdempotency
			}
			return *j, false, nil
		}
	}
	job.ID, job.Status, job.BatchesTotal = uuid.New(), repository.DraftJobQueued, len(batches)
	f.jobs[job.ID] = &job
	for _, b := range batches {
		b.JobID, b.Status, b.Excerpts = job.ID, repository.DraftBatchPending, len(b.ExcerptIDs)
		f.batches[job.ID] = append(f.batches[job.ID], b)
	}
	return job, true, nil
}

func (f *fakeJobs) Get(_ context.Context, _ string, id uuid.UUID) (repository.DraftJob, []repository.DraftJobBatch, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	j, ok := f.jobs[id]
	if !ok {
		return repository.DraftJob{}, nil, repository.ErrNotFound
	}
	return *j, append([]repository.DraftJobBatch{}, f.batches[id]...), nil
}

func (f *fakeJobs) List(context.Context, string, uuid.UUID, int) ([]repository.DraftJob, error) {
	return nil, nil
}

func (f *fakeJobs) Claim(_ context.Context, _ string, owner uuid.UUID, _ time.Duration) (repository.DraftJob, bool, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	for id, j := range f.jobs {
		if j.Status == repository.DraftJobQueued || (j.Status == repository.DraftJobRunning && f.owner[id] == uuid.Nil) {
			j.Status = repository.DraftJobRunning
			f.owner[id] = owner
			return *j, true, nil
		}
	}
	return repository.DraftJob{}, false, nil
}

func (f *fakeJobs) ExtendLease(_ context.Context, id, owner uuid.UUID, _ time.Duration) (bool, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.owner[id] == owner, nil
}

func (f *fakeJobs) batch(id uuid.UUID, idx int) *repository.DraftJobBatch {
	for i := range f.batches[id] {
		if f.batches[id][i].BatchIndex == idx {
			return &f.batches[id][i]
		}
	}
	return nil
}

func (f *fakeJobs) StartBatch(_ context.Context, id, owner uuid.UUID, idx int) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	b := f.batch(id, idx)
	if f.owner[id] != owner || b == nil || (b.Status != repository.DraftBatchPending && b.Status != repository.DraftBatchRunning) {
		return repository.ErrConflict
	}
	b.Status, b.Attempts, b.ErrorCode, b.ErrorMessage = repository.DraftBatchRunning, b.Attempts+1, nil, nil
	return nil
}

func (f *fakeJobs) EndBatch(_ context.Context, id, owner uuid.UUID, idx int, status string, result json.RawMessage, code, msg string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	b := f.batch(id, idx)
	if f.owner[id] != owner || b == nil || b.Status != repository.DraftBatchRunning {
		return repository.ErrConflict
	}
	b.Status = status
	if result != nil {
		b.Result = result
	}
	if code != "" {
		b.ErrorCode, b.ErrorMessage = &code, &msg
	}
	j := f.jobs[id]
	j.BatchesDone, j.BatchesFailed = 0, 0
	for _, x := range f.batches[id] {
		switch x.Status {
		case repository.DraftBatchSucceeded:
			j.BatchesDone++
		case repository.DraftBatchFailed:
			j.BatchesFailed++
		}
	}
	return nil
}

func (f *fakeJobs) Finish(_ context.Context, id, owner uuid.UUID, status string, draftID *uuid.UUID, code, msg string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.owner[id] != owner {
		return repository.ErrConflict
	}
	j := f.jobs[id]
	j.Status, j.DraftID = status, draftID
	j.ErrorCode, j.ErrorMessage = nil, nil
	if code != "" {
		j.ErrorCode, j.ErrorMessage = &code, &msg
	}
	f.owner[id] = uuid.Nil
	return nil
}

func (f *fakeJobs) Retry(_ context.Context, _ string, id uuid.UUID) (repository.DraftJob, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	j, ok := f.jobs[id]
	if !ok {
		return repository.DraftJob{}, repository.ErrNotFound
	}
	if j.Status != repository.DraftJobFailed {
		return repository.DraftJob{}, repository.ErrConflict
	}
	for i := range f.batches[id] {
		if f.batches[id][i].Status != repository.DraftBatchSucceeded {
			f.batches[id][i].Status, f.batches[id][i].Attempts = repository.DraftBatchPending, 0
		}
	}
	j.Status, j.BatchesFailed, j.ErrorCode = repository.DraftJobQueued, 0, nil
	return *j, nil
}

// fakeAI answers per batch_index with scripted partial Definitions.
type fakeAI struct {
	mu       sync.Mutex
	calls    map[int]int
	failFrom map[int]bool // batch_index -> always 504
}

func ev(pid, kind string, chunk int) EvidenceItem {
	ex := ldChunk(chunk)
	return EvidenceItem{ProposalID: pid, Kind: kind, ExcerptID: ex.ExcerptID.String(), Page: ex.PageStart, Quotation: ex.Content}
}

func (a *fakeAI) handler(t *testing.T) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req providerRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode: %v", err)
		}
		idx := 0
		if req.Context != nil {
			idx = req.Context.BatchIndex
			if !req.Context.Partial || req.Context.SectionTitle != "Thủ tục đăng ký khai sinh" {
				t.Errorf("bad batch context %+v", req.Context)
			}
		}
		for _, ex := range req.Excerpts {
			if ex.ChunkIndex != 0 { // ChunkIndex is not serialised
				t.Errorf("chunk index leaked to the AI request")
			}
			if strings.Contains(ex.Content, "đăng ký lại khai sinh Trình tự") || strings.Contains(ex.Content, "kết hôn") {
				t.Errorf("another procedure's text was sent to the model: %q", ex.Content)
			}
		}
		a.mu.Lock()
		a.calls[idx]++
		fail := a.failFrom[idx]
		a.mu.Unlock()
		if fail {
			w.WriteHeader(http.StatusGatewayTimeout)
			return
		}
		var def map[string]any
		var evidence []EvidenceItem
		switch idx {
		case 1:
			def = map[string]any{
				"procedure_code": "khai_sinh_v1", "name": "Đăng ký khai sinh",
				"intent_examples": []string{"Tôi muốn làm giấy khai sinh"},
				"slots": map[string]any{
					"hinh_thuc_nop": map[string]any{"type": "enum", "question": "Bạn nộp trực tiếp hay trực tuyến?", "enum_values": []string{"truc_tiep", "truc_tuyen"}},
				},
				"required_slots":    []string{"hinh_thuc_nop"}, // no required:* evidence on purpose
				"conditional_slots": []any{},
				"guidance": map[string]any{
					"summary": "Nộp hồ sơ tại Ủy ban nhân dân cấp xã.", "checklist": []string{"Tờ khai đăng ký khai sinh theo mẫu"},
					"fee_hint": "Miễn lệ phí đăng ký khai sinh đúng hạn",
				},
				"citations": []any{},
			}
			evidence = []EvidenceItem{
				ev("slot:hinh_thuc_nop", "slot", 3), ev("checklist:0", "checklist", 4),
				ev("guidance:fee_hint", "guidance", 5), ev("guidance:summary", "guidance", 2),
			}
		default:
			def = map[string]any{
				"procedure_code": "khai_sinh_v1", "name": "Đăng ký khai sinh",
				"intent_examples": []string{"tôi muốn làm giấy khai sinh", "Đăng ký khai sinh cho con"},
				"slots": map[string]any{
					"hinh_thuc_nop":      map[string]any{"type": "enum", "question": "Bạn nộp trực tiếp hay trực tuyến?", "enum_values": []string{"truc_tuyen", "truc_tiep"}},
					"co_giay_chung_sinh": map[string]any{"type": "boolean", "question": "Bạn có Giấy chứng sinh không?"},
				},
				"required_slots":    []string{"co_giay_chung_sinh"},
				"conditional_slots": []any{},
				"guidance": map[string]any{
					"summary": "Phần sau của thủ tục.", "checklist": []string{"Tờ khai đăng ký khai sinh theo mẫu.", "Giấy chứng sinh"},
					"fee_hint": "Lệ phí đăng ký khai sinh không đúng hạn là 8.000 đồng", "processing_time_hint": "Trong ngày làm việc",
				},
				"citations": []any{},
			}
			evidence = []EvidenceItem{
				ev("slot:hinh_thuc_nop", "slot", 43), ev("slot:co_giay_chung_sinh", "slot", 44),
				ev("required:co_giay_chung_sinh", "required_slot", 45),
				ev("checklist:0", "checklist", 46), ev("checklist:1", "checklist", 46),
				ev("guidance:fee_hint", "guidance", 47), ev("guidance:processing_time_hint", "guidance", 48),
			}
		}
		raw, _ := json.Marshal(def)
		_ = json.NewEncoder(w).Encode(ProviderResponse{
			SchemaVersion: "draft_definition.v1", ProcedureCode: "khai_sinh_v1", Definition: raw, Evidence: evidence,
			Provider: "fake", Model: "fake-model", PromptVersion: "draft.prompt.v2+batch.v1", LatencyMS: 10,
			Usage: map[string]any{"total_token_count": 100.0},
		})
	}
}

type ldEnv struct {
	svc    *Service
	runner *JobRunner
	jobs   *fakeJobs
	drafts *fakeDrafts
	ai     *fakeAI
}

func newLDEnv(t *testing.T) *ldEnv {
	t.Helper()
	ai := &fakeAI{calls: map[int]int{}, failFrom: map[int]bool{}}
	srv := httptest.NewServer(ai.handler(t))
	t.Cleanup(srv.Close)
	env := &ldEnv{jobs: newFakeJobs(), drafts: newFakeDrafts(), ai: ai}
	env.svc = &Service{
		XAID: "xa_test", Source: &fakeSource{all: ldExcerpts()}, Drafts: env.drafts,
		Client: NewClient(srv.URL, 5*time.Second), Jobs: env.jobs,
	}
	env.runner = NewJobRunner(env.svc)
	env.runner.Backoff = func(int) time.Duration { return 0 }
	env.runner.MaxAttempts = 2
	return env
}

func (e *ldEnv) createJob(t *testing.T, req DraftPlanRequest) DraftJobView {
	t.Helper()
	job, created, err := e.svc.CreateDraftJob(context.Background(), ldUser, uuid.New(), ldProc, req)
	if err != nil || !created {
		t.Fatalf("create job: created=%v err=%v", created, err)
	}
	return job
}

func (e *ldEnv) draft(t *testing.T, id uuid.UUID) DraftView {
	t.Helper()
	v, err := e.svc.Get(context.Background(), id)
	if err != nil {
		t.Fatal(err)
	}
	return v
}

func evidenceFor(v DraftView, pid string) *EvidenceItem {
	for i := range v.Evidence {
		if v.Evidence[i].ProposalID == pid {
			return &v.Evidence[i]
		}
	}
	return nil
}

func issueCodes(v DraftView) map[string]int {
	out := map[string]int{}
	for _, is := range v.Validation.Issues {
		out[is.Code]++
	}
	return out
}

// ---- tests ------------------------------------------------------------------

func TestDetectSectionsUsesHeadingsAndExactTitleOnly(t *testing.T) {
	secs := DetectSections(ldExcerpts())
	if len(secs) != 3 {
		t.Fatalf("sections = %+v", secs)
	}
	if secs[0].ChunkStart != 2 || secs[0].ChunkEnd != 48 || secs[1].ChunkStart != 49 || secs[2].ChunkEnd != 60 {
		t.Fatalf("boundaries = %+v", secs)
	}
	if secs[0].Title != "Thủ tục đăng ký khai sinh" || secs[0].Chunks != 47 {
		t.Fatalf("section 1 = %+v", secs[0])
	}
	if best := ScoreSections(secs, "Đăng ký khai sinh"); best != 0 {
		t.Fatalf("best = %d", best)
	}
	if secs[1].ExactMatch || secs[1].Match == 0 {
		t.Fatalf("'đăng ký lại khai sinh' must rank but never match: %+v", secs[1])
	}
	if best := ScoreSections(secs, "Đăng ký giám hộ"); best != -1 {
		t.Fatalf("unknown procedure must not match, got %d", best)
	}
	dup := append(append([]Section{}, secs...), Section{Title: "Thủ tục đăng ký khai sinh"})
	if best := ScoreSections(dup, "Đăng ký khai sinh"); best != -2 {
		t.Fatalf("two exact titles must be ambiguous, got %d", best)
	}
}

func TestDetectSectionsFlagsNumberingGap(t *testing.T) {
	items := ldExcerpts()
	items[56].Content = "5.Thủ tục đăng ký kết hôn Trình tự thực hiện: x"
	secs := DetectSections(items)
	if !secs[1].BoundaryUncertain || secs[0].BoundaryUncertain {
		t.Fatalf("2 -> 5 gap must flag section 2 only: %+v", secs)
	}
}

func TestPlanSelectsOnlyTheProcedureSection(t *testing.T) {
	env := newLDEnv(t)
	ctx := context.Background()
	plan, err := env.svc.PlanDraft(ctx, ldProc, DraftPlanRequest{DocumentID: ldDoc})
	if err != nil || !plan.Ready {
		t.Fatalf("plan err=%v plan=%+v", err, plan)
	}
	if plan.DocumentChunks != 61 || plan.SelectedChunks != 47 || plan.NewChunks != 47 {
		t.Fatalf("counts = %+v", plan)
	}
	if len(plan.ChunkRanges) != 1 || plan.ChunkRanges[0] != (ChunkRange{2, 48}) {
		t.Fatalf("ranges = %+v", plan.ChunkRanges)
	}
	if len(plan.Batches) != 2 || plan.Batches[0].Excerpts != BatchMaxExcerpts || plan.Batches[1].ChunkStart != 42 {
		t.Fatalf("batches = %+v", plan.Batches)
	}

	// A manual range spilling into "đăng ký lại khai sinh" is refused.
	mixed, err := env.svc.PlanDraft(ctx, ldProc, DraftPlanRequest{DocumentID: ldDoc, PageStart: 1, PageEnd: 12})
	if err != nil || mixed.Ready || mixed.BlockedReason != PlanMixedProcedures {
		t.Fatalf("mixed = %+v err=%v", mixed, err)
	}
	// A manual range covering only part of the section is allowed but warned.
	part, err := env.svc.PlanDraft(ctx, ldProc, DraftPlanRequest{DocumentID: ldDoc, PageStart: 2, PageEnd: 5})
	if err != nil || !part.Ready || !strings.Contains(strings.Join(part.Warnings, "|"), "chỉ gồm 20/47 đoạn") {
		t.Fatalf("partial = %+v err=%v", part, err)
	}
	// Choosing a non-matching section is allowed but warned.
	idx := 1
	other, err := env.svc.PlanDraft(ctx, ldProc, DraftPlanRequest{DocumentID: ldDoc, SectionIndex: &idx})
	if err != nil || !other.Ready || len(other.Warnings) == 0 {
		t.Fatalf("explicit section = %+v err=%v", other, err)
	}
}

func TestJobDraftsWholeSectionAndMergesBatches(t *testing.T) {
	env := newLDEnv(t)
	ctx := context.Background()
	job := env.createJob(t, DraftPlanRequest{DocumentID: ldDoc})
	if job.Status != repository.DraftJobQueued || job.BatchesTotal != 2 {
		t.Fatalf("job = %+v", job.DraftJob)
	}
	env.runner.Drain(ctx)
	got, _ := env.svc.GetDraftJob(ctx, job.ID)
	if got.Status != repository.DraftJobSucceeded || got.DraftID == nil || got.BatchesDone != 2 {
		t.Fatalf("job after run = %+v", got.DraftJob)
	}
	v := env.draft(t, *got.DraftID)
	var def aggDef
	if err := json.Unmarshal(v.Definition, &def); err != nil {
		t.Fatal(err)
	}

	// The same question (options in another order) merges; nothing from other procedures.
	if len(def.Slots) != 2 || strings.Join(def.Slots["hinh_thuc_nop"].EnumValues, ",") != "truc_tiep,truc_tuyen" {
		t.Fatalf("slots = %+v", def.Slots)
	}
	// Checklist de-duplicated across batches with evidence remapped to merged indexes.
	// Identical after normalising punctuation: merged, first wording kept.
	if strings.Join(def.Guidance.Checklist, "|") != "Tờ khai đăng ký khai sinh theo mẫu|Giấy chứng sinh" {
		t.Fatalf("checklist = %q", def.Guidance.Checklist)
	}
	if e := evidenceFor(v, "checklist:1"); e == nil || e.Quotation != ldToKhai2 {
		t.Fatalf("checklist:1 evidence = %+v", e)
	}
	if len(def.IntentExamples) != 2 {
		t.Fatalf("intent examples not de-duplicated: %q", def.IntentExamples)
	}

	// required:* evidence is never borrowed from the slot.
	if e := evidenceFor(v, "required:hinh_thuc_nop"); e != nil {
		t.Fatalf("required evidence was copied from the slot: %+v", e)
	}
	if e := evidenceFor(v, "required:co_giay_chung_sinh"); e == nil || e.Quotation != ldBatBuocGCS || !e.QuoteValid {
		t.Fatalf("required:co_giay_chung_sinh evidence = %+v", e)
	}

	// Different fee statements are a conflict: field left empty, not picked.
	if def.Guidance.FeeHint != "" || len(v.Conflicts) != 1 || v.Conflicts[0].ID != "guidance:fee_hint" || len(v.Conflicts[0].Candidates) != 2 {
		t.Fatalf("fee conflict = %q %+v", def.Guidance.FeeHint, v.Conflicts)
	}
	if c := v.Conflicts[0].Candidates[1]; c.Contributor != "batch:2" || c.Evidence == nil || c.Evidence.Quotation != ldPhi8k {
		t.Fatalf("candidate provenance = %+v", c)
	}
	if def.Guidance.ProcessingTimeHint != "Trong ngày làm việc" || def.Guidance.Summary != "Nộp hồ sơ tại Ủy ban nhân dân cấp xã." {
		t.Fatalf("guidance = %+v", def.Guidance)
	}

	// Provenance keeps every quotation; coverage reports the processed scope.
	var slotProv *ProposalProvenance
	for i := range v.Provenance {
		if v.Provenance[i].ProposalID == "slot:hinh_thuc_nop" {
			slotProv = &v.Provenance[i]
		}
	}
	if slotProv == nil || len(slotProv.Sources) != 2 || slotProv.Sources[1].ChunkIndex != 43 || !slotProv.Sources[1].QuoteValid {
		t.Fatalf("slot provenance = %+v", slotProv)
	}
	c := v.Coverage
	if c == nil || !c.ScopeComplete || !c.Complete || c.SelectedChunks != 47 || c.ProcessedChunks != 47 || c.Batches != 2 || c.DocumentChunks != 61 {
		t.Fatalf("coverage = %+v", c)
	}
	if len(v.ChunkRanges) != 1 || v.ChunkRanges[0] != (ChunkRange{2, 48}) || v.SourcePageStart != 1 || v.SourcePageEnd != 10 {
		t.Fatalf("source scope = %+v %d-%d", v.ChunkRanges, v.SourcePageStart, v.SourcePageEnd)
	}
	// The source has "Cách thức thực hiện" but no batch gave where_to_submit.
	if len(c.Gaps) != 1 || c.Gaps[0].Field != "guidance.where_to_submit" {
		t.Fatalf("gaps = %+v", c.Gaps)
	}
	codes := issueCodes(v)
	// Part I (chunk 0) names the procedure outside every section: scope is
	// complete, but legal-source coverage is not verified until reviewed.
	if c.OutsideMentionsReviewed || codes["OUTSIDE_SOURCES_UNREVIEWED"] != 1 {
		t.Fatalf("legal sources verified=%v codes=%+v", c.OutsideMentionsReviewed, codes)
	}
	if codes["SOURCE_CONFLICT"] != 1 || codes["COVERAGE_GAP"] != 1 || codes["SOURCE_PARTIALLY_PROCESSED"] != 0 {
		t.Fatalf("issues = %+v", codes)
	}
	for _, is := range v.Validation.Issues {
		if is.Code == "COVERAGE_GAP" && is.Severity != "error" {
			t.Fatalf("an unacknowledged coverage gap must block approval: %+v", is)
		}
	}
	if len(c.OutsideMentions) == 0 || c.OutsideMentions[0].Section != "" || c.OutsideMentions[0].Kind != "name" || c.OutsideMentions[0].PageStart != 1 || c.OutsideCriteria == "" {
		t.Fatalf("outside mentions = %+v", c.OutsideMentions)
	}
	if err := approvalGate(v); err == nil || !strings.Contains(err.Error(), "approval blocked") {
		t.Fatalf("approval must be blocked, got %v", err)
	}
	for _, w := range v.MissingRequirements {
		if !strings.HasPrefix(w, "[trang ") {
			t.Fatalf("missing requirement without page label: %q", w)
		}
	}
}

func TestResolveConflictUsesChosenCandidateEvidence(t *testing.T) {
	env := newLDEnv(t)
	ctx := context.Background()
	job := env.createJob(t, DraftPlanRequest{DocumentID: ldDoc})
	env.runner.Drain(ctx)
	got, _ := env.svc.GetDraftJob(ctx, job.ID)
	v := env.draft(t, *got.DraftID)

	one := 1
	stale := v.UpdatedAt.Add(-time.Minute)
	if _, err := env.svc.ResolveConflict(ctx, ldUser, v.ID, ResolveConflictRequest{ExpectedUpdatedAt: &stale, ConflictID: "guidance:fee_hint", Choice: &one}); err != ErrConflict {
		t.Fatalf("stale write must conflict, got %v", err)
	}
	bad := 5
	if _, err := env.svc.ResolveConflict(ctx, ldUser, v.ID, ResolveConflictRequest{ExpectedUpdatedAt: &v.UpdatedAt, ConflictID: "guidance:fee_hint", Choice: &bad}); err == nil {
		t.Fatal("out-of-range choice must fail")
	}
	out, err := env.svc.ResolveConflict(ctx, ldUser, v.ID, ResolveConflictRequest{ExpectedUpdatedAt: &v.UpdatedAt, ConflictID: "guidance:fee_hint", Choice: &one})
	if err != nil {
		t.Fatal(err)
	}
	var def aggDef
	_ = json.Unmarshal(out.Definition, &def)
	if def.Guidance.FeeHint != "Lệ phí đăng ký khai sinh không đúng hạn là 8.000 đồng" {
		t.Fatalf("fee = %q", def.Guidance.FeeHint)
	}
	if e := evidenceFor(out, "guidance:fee_hint"); e == nil || !e.QuoteValid || e.Quotation != ldPhi8k || e.ReviewStatus != ReviewPending {
		t.Fatalf("fee evidence = %+v", e)
	}
	if out.Conflicts[0].Status != ConflictResolved || *out.Conflicts[0].Chosen != 1 || out.Conflicts[0].ResolvedBy == nil {
		t.Fatalf("conflict = %+v", out.Conflicts[0])
	}
	if issueCodes(out)["SOURCE_CONFLICT"] != 0 {
		t.Fatalf("resolved conflict still reported: %+v", issueCodes(out))
	}
}

func TestPartialFailureKeepsFinishedBatchesAndRetriesOnlyFailed(t *testing.T) {
	env := newLDEnv(t)
	ctx := context.Background()
	env.ai.failFrom[2] = true
	job := env.createJob(t, DraftPlanRequest{DocumentID: ldDoc})
	env.runner.Drain(ctx)

	got, _ := env.svc.GetDraftJob(ctx, job.ID)
	if got.Status != repository.DraftJobFailed || got.DraftID != nil || got.ErrorCode == nil || *got.ErrorCode != "BATCH_FAILED" {
		t.Fatalf("job = %+v", got.DraftJob)
	}
	if got.Batches[0].Status != repository.DraftBatchSucceeded || got.Batches[1].Status != repository.DraftBatchFailed || got.Batches[1].Attempts != 2 {
		t.Fatalf("batches = %+v", got.Batches)
	}
	if *got.Batches[1].ErrorCode != "DRAFT_TIMEOUT" || env.ai.calls[2] != 2 || env.ai.calls[1] != 1 {
		t.Fatalf("batch 2 error=%v calls=%v", *got.Batches[1].ErrorCode, env.ai.calls)
	}
	if len(env.drafts.rows) != 0 {
		t.Fatal("no draft may be created from part of the source")
	}
	if _, err := env.svc.RetryDraftJob(ctx, uuid.New()); err == nil {
		t.Fatal("retry of unknown job must fail")
	}

	env.ai.failFrom[2] = false
	if _, err := env.svc.RetryDraftJob(ctx, job.ID); err != nil {
		t.Fatal(err)
	}
	env.runner.Drain(ctx)
	got, _ = env.svc.GetDraftJob(ctx, job.ID)
	if got.Status != repository.DraftJobSucceeded || got.DraftID == nil {
		t.Fatalf("job after retry = %+v", got.DraftJob)
	}
	if env.ai.calls[1] != 1 {
		t.Fatalf("finished batch 1 was re-run: calls=%v", env.ai.calls)
	}
	if _, err := env.svc.RetryDraftJob(ctx, job.ID); err != ErrJobNotRetryable {
		t.Fatalf("a SUCCEEDED job is not retryable, got %v", err)
	}
}

func TestAggregateRerunDoesNotDuplicateDraft(t *testing.T) {
	env := newLDEnv(t)
	ctx := context.Background()
	job := env.createJob(t, DraftPlanRequest{DocumentID: ldDoc})
	env.runner.Drain(ctx)
	got, _ := env.svc.GetDraftJob(ctx, job.ID)
	var plan DraftPlan
	_ = json.Unmarshal(got.Plan, &plan)
	again, err := env.svc.aggregateJob(ctx, got.DraftJob, plan, got.Batches)
	if err != nil || again != *got.DraftID || len(env.drafts.rows) != 1 {
		t.Fatalf("re-aggregate id=%s err=%v rows=%d", again, err, len(env.drafts.rows))
	}
}

func TestCreateJobReplaysSameRequestAndRejectsChangedBody(t *testing.T) {
	env := newLDEnv(t)
	ctx := context.Background()
	rid := uuid.New()
	first, created, err := env.svc.CreateDraftJob(ctx, ldUser, rid, ldProc, DraftPlanRequest{DocumentID: ldDoc})
	if err != nil || !created {
		t.Fatal(err)
	}
	replay, created, err := env.svc.CreateDraftJob(ctx, ldUser, rid, ldProc, DraftPlanRequest{DocumentID: ldDoc})
	if err != nil || created || replay.ID != first.ID {
		t.Fatalf("replay = %v created=%v err=%v", replay.ID, created, err)
	}
	idx := 0
	if _, _, err := env.svc.CreateDraftJob(ctx, ldUser, rid, ldProc, DraftPlanRequest{DocumentID: ldDoc, SectionIndex: &idx}); err != repository.ErrIdempotency {
		t.Fatalf("changed body must conflict, got %v", err)
	}
	if _, _, err := env.svc.CreateDraftJob(ctx, ldUser, uuid.New(), ldProc, DraftPlanRequest{DocumentID: ldDoc, PageStart: 1, PageEnd: 12}); err == nil || !strings.Contains(err.Error(), PlanMixedProcedures) {
		t.Fatalf("blocked plan must not create a job, got %v", err)
	}
}

// A SUPPLEMENT job adds the pages a base draft did not cover, keeps the base
// draft row unchanged and keeps Admin acceptances whose proposal did not change.
func TestSupplementMergesIntoNewDraftAndKeepsBase(t *testing.T) {
	env := newLDEnv(t)
	ctx := context.Background()
	baseDef, _ := json.Marshal(map[string]any{
		"procedure_code": "khai_sinh_v1", "name": "Đăng ký khai sinh",
		"slots": map[string]any{
			"co_giay_chung_sinh": map[string]any{"type": "boolean", "question": "Bạn có Giấy chứng sinh không?"},
			"noi_cu_tru":         map[string]any{"type": "string", "question": "Bạn cư trú ở đâu?"},
		},
		"required_slots": []string{}, "conditional_slots": []any{}, "intent_examples": []string{},
		"guidance": map[string]any{"summary": "Bản nền.", "checklist": []string{}}, "citations": []any{},
	})
	accepted := ev("slot:noi_cu_tru", "slot", 3)
	accepted.ReviewStatus = ReviewAccepted
	gcs := ev("slot:co_giay_chung_sinh", "slot", 4)
	gcs.ReviewStatus = ReviewAccepted
	payload, _ := json.Marshal(storedPayloadFromView(DraftView{
		Definition: baseDef, Evidence: []EvidenceItem{accepted, gcs}, GenerationID: ldGen, DocumentID: ldDoc,
		SourcePageStart: 1, SourcePageEnd: 8, ChunkRanges: []ChunkRange{{2, 41}},
		Warnings: []string{"[trang 1–8] cảnh báo nền"},
	}))
	base, _ := env.drafts.Insert(ctx, repository.ProcedureDraft{
		ID: uuid.New(), DocumentID: ldDoc, ProcedureID: ldProc, DraftDefinition: payload,
		Status: repository.DraftStatusDraft, CreatedBy: &ldUser, UpdatedBy: &ldUser,
	})
	before := string(env.drafts.rows[base.ID].DraftDefinition)

	plan, err := env.svc.PlanDraft(ctx, ldProc, DraftPlanRequest{DocumentID: ldDoc, Mode: JobModeSupplement, BaseDraftID: &base.ID})
	if err != nil || !plan.Ready || plan.NewChunks != 7 || len(plan.Batches) != 1 || plan.Batches[0].ChunkStart != 42 {
		t.Fatalf("supplement plan = %+v err=%v", plan, err)
	}
	job := env.createJob(t, DraftPlanRequest{DocumentID: ldDoc, Mode: JobModeSupplement, BaseDraftID: &base.ID})
	env.runner.Drain(ctx)
	got, _ := env.svc.GetDraftJob(ctx, job.ID)
	if got.Status != repository.DraftJobSucceeded {
		t.Fatalf("job = %+v", got.DraftJob)
	}
	v := env.draft(t, *got.DraftID)
	if v.ParentDraftID == nil || *v.ParentDraftID != base.ID || v.ID == base.ID {
		t.Fatalf("supplement must create a new child draft: %+v", v.ParentDraftID)
	}
	if string(env.drafts.rows[base.ID].DraftDefinition) != before {
		t.Fatal("base draft was modified")
	}
	var def aggDef
	_ = json.Unmarshal(v.Definition, &def)
	if _, ok := def.Slots["noi_cu_tru"]; !ok || len(def.Slots) != 3 {
		t.Fatalf("base slots lost: %+v", def.Slots)
	}
	if e := evidenceFor(v, "slot:noi_cu_tru"); e == nil || e.ReviewStatus != ReviewAccepted {
		t.Fatalf("unchanged base proposal must keep its acceptance: %+v", e)
	}
	if def.Guidance.Summary != "Bản nền." {
		t.Fatalf("base summary must win: %q", def.Guidance.Summary)
	}
	if v.Coverage.BaseChunks != 40 || v.Coverage.SelectedChunks != 7 || !v.Coverage.Complete {
		t.Fatalf("coverage = %+v", v.Coverage)
	}
	if len(v.ChunkRanges) != 1 || v.ChunkRanges[0] != (ChunkRange{2, 48}) {
		t.Fatalf("merged scope = %+v", v.ChunkRanges)
	}
	sort.Strings(v.Warnings)
	if v.Warnings[0] != "[trang 1–8] cảnh báo nền" {
		t.Fatalf("base warnings must not be re-labelled: %q", v.Warnings)
	}

	full, err := env.svc.PlanDraft(ctx, ldProc, DraftPlanRequest{DocumentID: ldDoc, Mode: JobModeSupplement, BaseDraftID: &v.ID})
	if err != nil || full.Ready || full.BlockedReason != PlanNothingNew {
		t.Fatalf("fully covered base = %+v err=%v", full, err)
	}
}

// Legal text is never matched fuzzily: one word ("không", a number) can flip
// the meaning, so only formatting-identical values merge.
func TestScalarValuesMergeOnlyWhenIdentical(t *testing.T) {
	mk := func(label, fee string) Contribution {
		def, _ := json.Marshal(map[string]any{
			"procedure_code": "p", "name": "n", "slots": map[string]any{},
			"guidance": map[string]any{"summary": "", "checklist": []string{fee}, "fee_hint": fee},
		})
		return Contribution{Label: label, PageStart: 1, PageEnd: 1, Definition: def}
	}
	conflicting := [][2]string{
		{"Không thu lệ phí đăng ký khai sinh tại xã", "Thu lệ phí đăng ký khai sinh tại xã"},
		{"Không được nộp hồ sơ trực tuyến", "Được nộp hồ sơ trực tuyến"},
		{"Lệ phí đăng ký khai sinh không đúng hạn tại Ủy ban nhân dân cấp xã là 8.000 đồng mỗi trường hợp",
			"Lệ phí đăng ký khai sinh không đúng hạn tại Ủy ban nhân dân cấp xã là 9.000 đồng mỗi trường hợp"},
		{"Miễn lệ phí đúng hạn", "Miễn lệ phí đúng hạn cho trẻ em"},
	}
	for _, pair := range conflicting {
		agg, err := Aggregate("p", "n", []Contribution{mk("batch:1", pair[0]), mk("batch:2", pair[1])}, nil)
		if err != nil {
			t.Fatal(err)
		}
		var def aggDef
		_ = json.Unmarshal(agg.Definition, &def)
		if len(agg.Conflicts) != 1 || def.Guidance.FeeHint != "" || len(agg.Conflicts[0].Candidates) != 2 {
			t.Fatalf("%q vs %q must stay an open conflict: fee=%q %+v", pair[0], pair[1], def.Guidance.FeeHint, agg.Conflicts)
		}
		if len(def.Guidance.Checklist) != 2 {
			t.Fatalf("%q vs %q must stay two checklist items: %q", pair[0], pair[1], def.Guidance.Checklist)
		}
	}
	agg, _ := Aggregate("p", "n", []Contribution{mk("batch:1", "Miễn lệ phí, đúng hạn."), mk("batch:2", "miễn  lệ phí đúng hạn")}, nil)
	var def aggDef
	_ = json.Unmarshal(agg.Definition, &def)
	if len(agg.Conflicts) != 0 || def.Guidance.FeeHint != "Miễn lệ phí, đúng hạn." || len(def.Guidance.Checklist) != 1 {
		t.Fatalf("case/punctuation-only differences should merge: %q %q %+v", def.Guidance.FeeHint, def.Guidance.Checklist, agg.Conflicts)
	}
}

func TestSlotTypeDisagreementIsAConflictTheAdminResolves(t *testing.T) {
	env := newLDEnv(t)
	ctx := context.Background()
	mk := func(label, typ string, enum []string, chunk int) Contribution {
		slot := map[string]any{"type": typ, "question": "Hình thức nộp?"}
		if enum != nil {
			slot["enum_values"] = enum
		}
		def, _ := json.Marshal(map[string]any{
			"procedure_code": "khai_sinh_v1", "name": "Đăng ký khai sinh", "slots": map[string]any{"hinh_thuc_nop": slot},
			"guidance": map[string]any{"summary": "", "checklist": []string{}},
		})
		return Contribution{Label: label, PageStart: 1, PageEnd: 2, Definition: def, Evidence: []EvidenceItem{ev("slot:hinh_thuc_nop", "slot", chunk)}}
	}
	agg, err := Aggregate("khai_sinh_v1", "Đăng ký khai sinh", []Contribution{
		mk("batch:1", "string", nil, 3), mk("batch:2", "enum", []string{"truc_tiep", "buu_chinh"}, 43), mk("batch:3", "string", nil, 4),
	}, ldExcerpts())
	if err != nil {
		t.Fatal(err)
	}
	if len(agg.Conflicts) != 1 || agg.Conflicts[0].Field != "slots.hinh_thuc_nop" || len(agg.Conflicts[0].Candidates) != 2 {
		t.Fatalf("slot type conflict = %+v", agg.Conflicts)
	}
	if agg.Conflicts[0].Candidates[1].Value != "enum: Hình thức nộp? [truc_tiep, buu_chinh]" || agg.Conflicts[0].Candidates[1].Evidence.Quotation != ldBuuChinh {
		t.Fatalf("candidate = %+v", agg.Conflicts[0].Candidates[1])
	}

	// Store a draft carrying the conflict, then resolve to the enum candidate.
	view := DraftView{
		Definition: agg.Definition, Evidence: agg.Evidence, Conflicts: agg.Conflicts,
		GenerationID: ldGen, DocumentID: ldDoc, ChunkRanges: []ChunkRange{{2, 48}},
	}
	payload, _ := json.Marshal(storedPayloadFromView(view))
	row, _ := env.drafts.Insert(ctx, repository.ProcedureDraft{ID: uuid.New(), DocumentID: ldDoc, ProcedureID: ldProc, DraftDefinition: payload, Status: repository.DraftStatusDraft})
	if err := approvalGate(env.draft(t, row.ID)); err == nil {
		t.Fatal("open slot conflict must block approval")
	}
	one := 1
	out, err := env.svc.ResolveConflict(ctx, ldUser, row.ID, ResolveConflictRequest{ExpectedUpdatedAt: &row.UpdatedAt, ConflictID: "slot:hinh_thuc_nop", Choice: &one})
	if err != nil {
		t.Fatal(err)
	}
	var def aggDef
	_ = json.Unmarshal(out.Definition, &def)
	if def.Slots["hinh_thuc_nop"].Type != "enum" || len(def.Slots["hinh_thuc_nop"].EnumValues) != 2 {
		t.Fatalf("slot after resolve = %+v", def.Slots)
	}
	if e := evidenceFor(out, "slot:hinh_thuc_nop"); e == nil || e.Quotation != ldBuuChinh || !e.QuoteValid {
		t.Fatalf("slot evidence after resolve = %+v", e)
	}
}

func TestCoverageGapBlocksUntilAcknowledgedWithReason(t *testing.T) {
	env := newLDEnv(t)
	ctx := context.Background()
	job := env.createJob(t, DraftPlanRequest{DocumentID: ldDoc})
	env.runner.Drain(ctx)
	got, _ := env.svc.GetDraftJob(ctx, job.ID)
	v := env.draft(t, *got.DraftID)
	if _, err := env.svc.AcknowledgeGap(ctx, ldUser, v.ID, AcknowledgeGapRequest{ExpectedUpdatedAt: &v.UpdatedAt, Field: "guidance.where_to_submit", Reason: "x"}); err == nil {
		t.Fatal("a reason is required")
	}
	if _, err := env.svc.AcknowledgeGap(ctx, ldUser, v.ID, AcknowledgeGapRequest{ExpectedUpdatedAt: &v.UpdatedAt, Field: "guidance.fee_hint", Reason: "không có"}); err == nil {
		t.Fatal("only a current gap can be acknowledged")
	}
	out, err := env.svc.AcknowledgeGap(ctx, ldUser, v.ID, AcknowledgeGapRequest{
		ExpectedUpdatedAt: &v.UpdatedAt, Field: "guidance.where_to_submit", Reason: "Mục Cách thức chỉ nêu hình thức nộp, nơi nộp đã có trong tóm tắt.",
	})
	if err != nil {
		t.Fatal(err)
	}
	codes := issueCodes(out)
	if codes["COVERAGE_GAP"] != 0 || codes["COVERAGE_GAP_ACKNOWLEDGED"] != 1 {
		t.Fatalf("issues after acknowledge = %+v", codes)
	}
	if g := out.Coverage.Gaps; len(g) != 1 || !g[0].Acknowledged || g[0].Reason == "" || out.Coverage.Acknowledged[0].By != ldUser {
		t.Fatalf("gaps = %+v acks = %+v", g, out.Coverage.Acknowledged)
	}
}

func TestRepeatedHeadingInAppendixIsAmbiguousNotSwallowed(t *testing.T) {
	items := ldExcerpts()
	items[58].Content = "PHỤ LỤC II 1.Thủ tục đăng ký khai sinh Trình tự thực hiện: nội dung khác."
	secs := DetectSections(items)
	if len(secs) != 4 || secs[3].ChunkStart != 58 || secs[2].ChunkEnd != 57 {
		t.Fatalf("appendix section must be kept separately: %+v", secs)
	}
	if best := ScoreSections(secs, "Đăng ký khai sinh"); best != -2 {
		t.Fatalf("two sections with the procedure title must be ambiguous, got %d", best)
	}
	// The overlap copy of a heading in the next chunk is still one section.
	items = ldExcerpts()
	items[3].Content = ldHeading1
	if secs := DetectSections(items); len(secs) != 3 || secs[0].ChunkStart != 2 {
		t.Fatalf("overlap duplicate must not open a section: %+v", secs)
	}
	env := newLDEnv(t)
	env.svc.Source = &fakeSource{all: func() []Excerpt {
		x := ldExcerpts()
		x[58].Content = "PHỤ LỤC II 1.Thủ tục đăng ký khai sinh Trình tự thực hiện: khác."
		return x
	}()}
	plan, err := env.svc.PlanDraft(context.Background(), ldProc, DraftPlanRequest{DocumentID: ldDoc})
	if err != nil || plan.Ready || plan.BlockedReason != PlanSectionAmbiguous {
		t.Fatalf("plan = %+v err=%v", plan.BlockedReason, err)
	}
}

func TestRangeOutsideOwnSection(t *testing.T) {
	env := newLDEnv(t)
	ctx := context.Background()
	// Page 13 is the tail of another section: none of the procedure's own chunks.
	full, err := env.svc.PlanDraft(ctx, ldProc, DraftPlanRequest{DocumentID: ldDoc, PageStart: 13, PageEnd: 13})
	if err != nil || full.Ready || full.BlockedReason != PlanOutsideSection {
		t.Fatalf("FULL outside own section = %+v err=%v", full.BlockedReason, err)
	}
	// SUPPLEMENT: allowed with a warning, and the still-undrafted part of the
	// section is named.
	baseDef, _ := json.Marshal(map[string]any{"procedure_code": "khai_sinh_v1", "name": "Đăng ký khai sinh", "slots": map[string]any{}, "guidance": map[string]any{"summary": "", "checklist": []string{}}})
	payload, _ := json.Marshal(storedPayloadFromView(DraftView{Definition: baseDef, GenerationID: ldGen, DocumentID: ldDoc, ChunkRanges: []ChunkRange{{10, 40}}}))
	base, _ := env.drafts.Insert(ctx, repository.ProcedureDraft{ID: uuid.New(), DocumentID: ldDoc, ProcedureID: ldProc, DraftDefinition: payload, Status: repository.DraftStatusDraft})
	sup, err := env.svc.PlanDraft(ctx, ldProc, DraftPlanRequest{DocumentID: ldDoc, Mode: JobModeSupplement, BaseDraftID: &base.ID, PageStart: 1, PageEnd: 1})
	if err != nil || !sup.Ready {
		t.Fatalf("supplement = %+v err=%v", sup, err)
	}
	joined := strings.Join(sup.Warnings, "|")
	if !strings.Contains(joined, "vẫn chưa phủ") {
		t.Fatalf("uncovered part of the section must be named: %q", sup.Warnings)
	}
	if _, err := (&SourceStore{}).Resolve(ctx, "xa_test", ldProc, ldDoc, ldGen, 0, 0); err == nil || !strings.Contains(err.Error(), "invalid page range") {
		t.Fatalf("a missing page range must be rejected, not defaulted: %v", err)
	}
}

func TestSplitBatchesRespectsLimitsAndGaps(t *testing.T) {
	items := ldExcerpts()
	sel := append(append([]Excerpt{}, items[2:10]...), items[20:25]...)
	b := splitBatches(sel)
	if len(b) != 2 || b[0].ChunkEnd != 9 || b[1].ChunkStart != 20 || b[1].Excerpts != 5 {
		t.Fatalf("non-contiguous chunks must start a new batch: %+v", b)
	}
	if got := chunkRangesOf([]int{5, 3, 4, 9, 10, 12}); len(got) != 3 || got[0] != (ChunkRange{3, 5}) || got[2] != (ChunkRange{12, 12}) {
		t.Fatalf("ranges = %+v", got)
	}
}

func TestSameSlotKeyWithDifferentQuestionOrOptionsIsAConflict(t *testing.T) {
	mk := func(label, question string, enum []string) Contribution {
		def, _ := json.Marshal(map[string]any{
			"procedure_code": "p", "name": "n",
			"slots":    map[string]any{"hinh_thuc_nop": map[string]any{"type": "enum", "question": question, "enum_values": enum}},
			"guidance": map[string]any{"summary": "", "checklist": []string{}},
		})
		return Contribution{Label: label, Definition: def}
	}
	cases := []struct {
		name string
		b    Contribution
		want int
	}{
		{"different question", mk("batch:2", "Bạn nộp theo hình thức nào?", []string{"truc_tiep", "truc_tuyen"}), 1},
		{"different options", mk("batch:2", "Bạn nộp trực tiếp hay trực tuyến?", []string{"truc_tiep", "buu_chinh"}), 1},
		{"same, other order and case", mk("batch:2", "bạn nộp trực tiếp hay trực tuyến", []string{"Truc_tuyen", "truc_tiep"}), 0},
	}
	for _, tc := range cases {
		agg, err := Aggregate("p", "n", []Contribution{mk("batch:1", "Bạn nộp trực tiếp hay trực tuyến?", []string{"truc_tiep", "truc_tuyen"}), tc.b}, nil)
		if err != nil {
			t.Fatal(err)
		}
		var def aggDef
		_ = json.Unmarshal(agg.Definition, &def)
		if len(agg.Conflicts) != tc.want || strings.Join(def.Slots["hinh_thuc_nop"].EnumValues, ",") != "truc_tiep,truc_tuyen" {
			t.Fatalf("%s: conflicts=%+v slot=%+v (options must never be unioned)", tc.name, agg.Conflicts, def.Slots)
		}
	}
}

func TestScoreSectionsMarksEveryExactMatch(t *testing.T) {
	secs := []Section{{Title: "Thủ tục đăng ký khai sinh"}, {Title: "Thủ tục đăng ký kết hôn"}, {Title: "Thủ tục đăng ký khai sinh"}, {Title: "Thủ tục đăng ký khai sinh"}}
	if best := ScoreSections(secs, "Đăng ký khai sinh"); best != -2 {
		t.Fatalf("best = %d", best)
	}
	for i, want := range []bool{true, false, true, true} {
		if secs[i].ExactMatch != want {
			t.Fatalf("section %d exact=%v, want %v", i, secs[i].ExactMatch, want)
		}
	}
}

// storeJobDraft stores an aggregate as a job draft so findings can be exercised.
func storeJobDraft(t *testing.T, env *ldEnv, agg AggregateResult, mentions []OutsideMention) repository.ProcedureDraft {
	t.Helper()
	view := DraftView{
		Definition: agg.Definition, Evidence: agg.Evidence, Conflicts: agg.Conflicts,
		GenerationID: ldGen, DocumentID: ldDoc, ChunkRanges: []ChunkRange{{2, 48}},
		Coverage: &DraftCoverage{ScopeComplete: true, Complete: true, Truncated: agg.Truncated, OutsideMentions: mentions},
	}
	applyAggregateFindings(&view, ldExcerpts()[2:49])
	payload, _ := json.Marshal(storedPayloadFromView(view))
	validation, _ := json.Marshal(view.Validation)
	row, err := env.drafts.Insert(context.Background(), repository.ProcedureDraft{
		ID: uuid.New(), DocumentID: ldDoc, ProcedureID: ldProc, DraftDefinition: payload, ValidationResult: validation, Status: repository.DraftStatusDraft,
	})
	if err != nil {
		t.Fatal(err)
	}
	return row
}

func TestTruncatedItemsAreRecordedAndBlockUntilReviewed(t *testing.T) {
	env := newLDEnv(t)
	ctx := context.Background()
	items := func(from, to int) []string {
		out := []string{}
		for i := from; i < to; i++ {
			out = append(out, fmt.Sprintf("Giấy tờ số %d", i))
		}
		return out
	}
	mk := func(label string, list []string) Contribution {
		def, _ := json.Marshal(map[string]any{"procedure_code": "khai_sinh_v1", "name": "Đăng ký khai sinh", "slots": map[string]any{},
			"guidance": map[string]any{"summary": "", "checklist": list}})
		return Contribution{Label: label, Definition: def}
	}
	agg, err := Aggregate("khai_sinh_v1", "Đăng ký khai sinh", []Contribution{mk("batch:1", items(0, 30)), mk("batch:2", items(25, 45))}, nil)
	if err != nil {
		t.Fatal(err)
	}
	var def aggDef
	_ = json.Unmarshal(agg.Definition, &def)
	if len(def.Guidance.Checklist) != aggMaxChecklist || len(agg.Truncated) != 1 {
		t.Fatalf("checklist=%d truncated=%+v", len(def.Guidance.Checklist), agg.Truncated)
	}
	tr := agg.Truncated[0]
	if tr.Field != "guidance.checklist" || !tr.Blocking || strings.Join(tr.Dropped, "|") != "Giấy tờ số 40|Giấy tờ số 41|Giấy tờ số 42|Giấy tờ số 43|Giấy tờ số 44" {
		t.Fatalf("truncation = %+v", tr)
	}
	row := storeJobDraft(t, env, agg, nil)
	v := env.draft(t, row.ID)
	if issueCodes(v)["CONTENT_TRUNCATED"] != 1 || approvalGate(v) == nil {
		t.Fatalf("truncation must block: %+v", issueCodes(v))
	}
	out, err := env.svc.AcknowledgeGap(ctx, ldUser, row.ID, AcknowledgeGapRequest{
		ExpectedUpdatedAt: &v.UpdatedAt, Field: "truncated:guidance.checklist", Reason: "Đã đối chiếu 5 mục bị cắt, trùng nghĩa với các mục đã có.",
	})
	if err != nil {
		t.Fatal(err)
	}
	if c := issueCodes(out); c["CONTENT_TRUNCATED"] != 0 || c["CONTENT_TRUNCATED_ACKNOWLEDGED"] != 1 {
		t.Fatalf("after review = %+v", c)
	}
}

// An acknowledgement applies to the exact state it was given for: once the
// field is filled it is dropped, and emptying the field again reopens the gap.
func TestGapAcknowledgementDoesNotSurviveContentChanges(t *testing.T) {
	env := newLDEnv(t)
	ctx := context.Background()
	job := env.createJob(t, DraftPlanRequest{DocumentID: ldDoc})
	env.runner.Drain(ctx)
	got, _ := env.svc.GetDraftJob(ctx, job.ID)
	v := env.draft(t, *got.DraftID)
	v, err := env.svc.AcknowledgeGap(ctx, ldUser, v.ID, AcknowledgeGapRequest{ExpectedUpdatedAt: &v.UpdatedAt, Field: "guidance.where_to_submit", Reason: "Văn bản không nêu nơi nộp riêng."})
	if err != nil || len(v.Coverage.Acknowledged) != 1 || v.Coverage.Acknowledged[0].Basis == "" {
		t.Fatalf("ack err=%v acks=%+v", err, v.Coverage.Acknowledged)
	}
	setWhere := func(view DraftView, value string) DraftView {
		var obj map[string]any
		_ = json.Unmarshal(view.Definition, &obj)
		obj["guidance"].(map[string]any)["where_to_submit"] = value
		raw, _ := json.Marshal(obj)
		out, err := env.svc.Patch(ctx, ldUser, view.ID, PatchRequest{Definition: raw, ExpectedUpdatedAt: &view.UpdatedAt})
		if err != nil {
			t.Fatal(err)
		}
		return out
	}
	v = setWhere(v, "Ủy ban nhân dân cấp xã")
	if len(v.Coverage.Acknowledged) != 0 || len(v.Coverage.Gaps) != 0 {
		t.Fatalf("filled field must drop the acknowledgement: %+v %+v", v.Coverage.Acknowledged, v.Coverage.Gaps)
	}
	v = setWhere(v, "")
	if len(v.Coverage.Gaps) != 1 || v.Coverage.Gaps[0].Acknowledged || issueCodes(v)["COVERAGE_GAP"] != 1 {
		t.Fatalf("emptied field must reopen the gap: %+v", v.Coverage.Gaps)
	}
}

func TestOutsideMentionsReviewVerifiesLegalSources(t *testing.T) {
	env := newLDEnv(t)
	ctx := context.Background()
	agg, _ := Aggregate("khai_sinh_v1", "Đăng ký khai sinh", nil, nil)
	mentions := []OutsideMention{{Kind: "name", Section: "", PageStart: 1, PageEnd: 1, Chunks: 1}, {Kind: "name", Section: "Thủ tục đăng ký lại khai sinh", PageStart: 10, PageEnd: 12, Chunks: 3}}
	row := storeJobDraft(t, env, agg, mentions)
	v := env.draft(t, row.ID)
	if v.Coverage.OutsideMentionsReviewed || issueCodes(v)["OUTSIDE_SOURCES_UNREVIEWED"] != 1 {
		t.Fatalf("unreviewed outside mention must block: %+v", issueCodes(v))
	}
	out, err := env.svc.AcknowledgeGap(ctx, ldUser, row.ID, AcknowledgeGapRequest{ExpectedUpdatedAt: &v.UpdatedAt, Field: "outside_mentions", Reason: "Trang 1 là bảng danh mục, không có quy định riêng."})
	if err != nil {
		t.Fatal(err)
	}
	if !out.Coverage.OutsideMentionsReviewed || issueCodes(out)["OUTSIDE_SOURCES_UNREVIEWED"] != 0 {
		t.Fatalf("after review = %+v", issueCodes(out))
	}
	// Mentions only inside other procedures' sections need no review.
	row2 := storeJobDraft(t, env, agg, mentions[1:])
	if v2 := env.draft(t, row2.ID); !v2.Coverage.OutsideMentionsReviewed {
		t.Fatal("mentions inside other procedure sections are not unreviewed legal sources")
	}
}

// Choosing the second candidate moves the proposal's provenance to that
// candidate's quotations, so Definition, evidence and provenance agree.
func TestResolveConflictRebuildsProvenance(t *testing.T) {
	env := newLDEnv(t)
	ctx := context.Background()
	mk := func(label, typ string, enum []string, chunks ...int) Contribution {
		slot := map[string]any{"type": typ, "question": "Hình thức nộp?"}
		if enum != nil {
			slot["enum_values"] = enum
		}
		def, _ := json.Marshal(map[string]any{"procedure_code": "khai_sinh_v1", "name": "Đăng ký khai sinh",
			"slots": map[string]any{"hinh_thuc_nop": slot}, "guidance": map[string]any{"summary": "", "checklist": []string{}}})
		evs := []EvidenceItem{}
		for _, ch := range chunks {
			evs = append(evs, ev("slot:hinh_thuc_nop", "slot", ch))
		}
		return Contribution{Label: label, PageStart: 1, PageEnd: 9, Definition: def, Evidence: evs}
	}
	agg, err := Aggregate("khai_sinh_v1", "Đăng ký khai sinh", []Contribution{
		mk("batch:1", "string", nil, 3), mk("batch:2", "enum", []string{"truc_tiep", "buu_chinh"}, 43), mk("batch:3", "enum", []string{"buu_chinh", "truc_tiep"}, 44),
	}, ldExcerpts())
	if err != nil {
		t.Fatal(err)
	}
	cands := agg.Conflicts[0].Candidates
	if len(cands) != 2 || len(cands[0].Sources) != 1 || len(cands[1].Sources) != 2 || cands[1].Sources[1].ChunkIndex != 44 {
		t.Fatalf("candidate sources = %+v", cands)
	}
	row := storeJobDraft(t, env, agg, nil)
	v := env.draft(t, row.ID)
	v.Provenance = agg.Provenance
	payload, _ := json.Marshal(storedPayloadFromView(v))
	env.drafts.rows[row.ID] = repository.ProcedureDraft{ID: row.ID, DocumentID: ldDoc, ProcedureID: ldProc, DraftDefinition: payload,
		ValidationResult: env.drafts.rows[row.ID].ValidationResult, Status: repository.DraftStatusDraft, UpdatedAt: row.UpdatedAt, ProcedureCode: "khai_sinh_v1"}
	one := 1
	out, err := env.svc.ResolveConflict(ctx, ldUser, row.ID, ResolveConflictRequest{ExpectedUpdatedAt: &row.UpdatedAt, ConflictID: "slot:hinh_thuc_nop", Choice: &one})
	if err != nil {
		t.Fatal(err)
	}
	var prov *ProposalProvenance
	for i := range out.Provenance {
		if out.Provenance[i].ProposalID == "slot:hinh_thuc_nop" {
			prov = &out.Provenance[i]
		}
	}
	var current MaterialProposal
	for _, p := range out.Proposals {
		if p.ID == "slot:hinh_thuc_nop" {
			current = p
		}
	}
	if prov == nil || prov.Content != current.Content || len(prov.Sources) != 2 || prov.Sources[0].ChunkIndex != 43 {
		t.Fatalf("provenance after resolve = %+v (proposal %q)", prov, current.Content)
	}
	if e := evidenceFor(out, "slot:hinh_thuc_nop"); e == nil || e.Quotation != ldBuuChinh {
		t.Fatalf("evidence after resolve = %+v", e)
	}
}

func TestCitationTruncationBlocks(t *testing.T) {
	cites := func(from, to int) []any {
		out := []any{}
		for i := from; i < to; i++ {
			out = append(out, map[string]any{"doc_id": "d", "title": fmt.Sprintf("Văn bản %d", i), "source_type": "ai_draft_unapproved"})
		}
		return out
	}
	mk := func(label string, c []any) Contribution {
		def, _ := json.Marshal(map[string]any{"procedure_code": "p", "name": "n", "slots": map[string]any{},
			"guidance": map[string]any{"summary": "", "checklist": []string{}}, "citations": c})
		return Contribution{Label: label, Definition: def}
	}
	agg, _ := Aggregate("p", "n", []Contribution{mk("batch:1", cites(0, 8)), mk("batch:2", cites(5, 15))}, nil)
	if len(agg.Truncated) != 1 || agg.Truncated[0].Field != "citations" || !agg.Truncated[0].Blocking || len(agg.Truncated[0].Dropped) != 5 {
		t.Fatalf("citation truncation = %+v", agg.Truncated)
	}
}

func TestUncertainBoundaryBlocksUntilConfirmed(t *testing.T) {
	env := newLDEnv(t)
	ctx := context.Background()
	agg, _ := Aggregate("khai_sinh_v1", "Đăng ký khai sinh", nil, nil)
	view := DraftView{
		Definition: agg.Definition, GenerationID: ldGen, DocumentID: ldDoc, ChunkRanges: []ChunkRange{{2, 48}},
		Coverage: &DraftCoverage{ScopeComplete: true, Complete: true, BoundaryUncertain: true, SectionTitle: "Thủ tục đăng ký khai sinh", PageStart: 1, PageEnd: 10},
	}
	applyAggregateFindings(&view, ldExcerpts()[2:49])
	payload, _ := json.Marshal(storedPayloadFromView(view))
	validation, _ := json.Marshal(view.Validation)
	row, _ := env.drafts.Insert(ctx, repository.ProcedureDraft{ID: uuid.New(), DocumentID: ldDoc, ProcedureID: ldProc, DraftDefinition: payload, ValidationResult: validation, Status: repository.DraftStatusDraft})
	v := env.draft(t, row.ID)
	if issueCodes(v)["SECTION_BOUNDARY_UNCERTAIN"] != 1 || v.Coverage.BoundaryReviewed {
		t.Fatalf("uncertain boundary must block: %+v", issueCodes(v))
	}
	out, err := env.svc.AcknowledgeGap(ctx, ldUser, row.ID, AcknowledgeGapRequest{ExpectedUpdatedAt: &v.UpdatedAt, Field: "section_boundary", Reason: "Đã xem trang 10: hết mục khai sinh."})
	if err != nil {
		t.Fatal(err)
	}
	if issueCodes(out)["SECTION_BOUNDARY_UNCERTAIN"] != 0 || !out.Coverage.BoundaryReviewed {
		t.Fatalf("after confirm = %+v", issueCodes(out))
	}
}

func TestOutsideSearchFindsGeneralRulesOutsideSections(t *testing.T) {
	items := ldExcerpts()
	items[1].Content = "PHẦN II. Quy định chung: thời hạn giải quyết tính theo ngày làm việc."
	items[0].Content = "Danh mục không liên quan."
	secs := DetectSections(items)
	ms := outsideMentions(secs, items, "Đăng ký khai sinh", []ChunkRange{{2, 48}})
	if len(ms) == 0 || ms[0].Kind != "general" || ms[0].Section != "" || ms[0].Chunks != 1 {
		t.Fatalf("general-rule chunk outside sections must be listed: %+v", ms)
	}
	for _, m := range ms {
		if m.Kind == "general" && m.Section != "" {
			t.Fatalf("general keywords inside another procedure's section are not listed: %+v", m)
		}
	}
	if !generalRuleRe.MatchString("nộp phí cấp bản sao") || generalRuleRe.MatchString("phíaa") {
		t.Fatal("fee keyword must match on Vietnamese word boundaries")
	}
}

// Pages 2, 100, 450 and 2, 150, 450 are different places: they must be kept
// apart for the Admin and give different review bases.
func TestOutsideMentionsKeepExactPagesAndChunks(t *testing.T) {
	mkDoc := func(pages ...int) []Excerpt {
		out := []Excerpt{}
		for i, p := range pages {
			content := "Quy định chung về lệ phí."
			sum := sha256.Sum256([]byte(content))
			// In one document a chunk index identifies a place; use page*7 here.
			_ = i
			out = append(out, Excerpt{ExcerptID: uuid.New(), ChunkIndex: p * 7, PageStart: p, PageEnd: p, Content: content, TextSHA256: hex.EncodeToString(sum[:])})
		}
		return out
	}
	a := outsideMentions(nil, mkDoc(2, 100, 450), "Đăng ký khai sinh", nil)
	b := outsideMentions(nil, mkDoc(2, 150, 450), "Đăng ký khai sinh", nil)
	if len(a) != 1 || len(a[0].Pages) != 3 || a[0].Pages[1] != (PageSpan{100, 100}) || len(a[0].ChunkIndexes) != 3 {
		t.Fatalf("exact pages lost: %+v", a)
	}
	if a[0].PageStart != b[0].PageStart || a[0].PageEnd != b[0].PageEnd || a[0].Chunks != b[0].Chunks {
		t.Fatalf("setup: both sets must share bounds and count: %+v %+v", a, b)
	}
	if outsideBasis(a) == outsideBasis(b) {
		t.Fatal("same bounds and count but different chunks must give different review bases")
	}
	c := outsideMentions(nil, mkDoc(7, 8, 9), "Đăng ký khai sinh", nil)
	if len(c[0].Pages) != 1 || c[0].Pages[0] != (PageSpan{7, 9}) {
		t.Fatalf("adjacent pages should form one span: %+v", c[0].Pages)
	}
}

// A slot option's main evidence comes from any batch that proposed it with a
// valid quotation, not only from the first one.
func TestSlotCandidateEvidenceUsesFirstValidQuote(t *testing.T) {
	mk := func(label, typ string, enum []string, e EvidenceItem) Contribution {
		slot := map[string]any{"type": typ, "question": "Hình thức nộp?"}
		if enum != nil {
			slot["enum_values"] = enum
		}
		def, _ := json.Marshal(map[string]any{"procedure_code": "p", "name": "n", "slots": map[string]any{"hinh_thuc_nop": slot},
			"guidance": map[string]any{"summary": "", "checklist": []string{}}})
		return Contribution{Label: label, Definition: def, Evidence: []EvidenceItem{e}}
	}
	bad := ev("slot:hinh_thuc_nop", "slot", 43)
	bad.Quotation = "câu không có trong tài liệu"
	agg, err := Aggregate("p", "n", []Contribution{
		mk("batch:1", "string", nil, ev("slot:hinh_thuc_nop", "slot", 3)),
		mk("batch:2", "enum", []string{"truc_tiep", "buu_chinh"}, bad),
		mk("batch:3", "enum", []string{"truc_tiep", "buu_chinh"}, ev("slot:hinh_thuc_nop", "slot", 43)),
	}, ldExcerpts())
	if err != nil {
		t.Fatal(err)
	}
	cand := agg.Conflicts[0].Candidates[1]
	if cand.Evidence == nil || cand.Evidence.Quotation != ldBuuChinh || len(cand.Sources) != 2 || cand.Sources[0].QuoteValid {
		t.Fatalf("candidate = %+v", cand)
	}
}

func TestKeepCitationSwapsAndAppends(t *testing.T) {
	env := newLDEnv(t)
	ctx := context.Background()
	cites := func(from, to int) []any {
		out := []any{}
		for i := from; i < to; i++ {
			out = append(out, map[string]any{"doc_id": "d", "title": fmt.Sprintf("Văn bản %d", i), "source_type": "ai_draft_unapproved"})
		}
		return out
	}
	mk := func(label string, c []any) Contribution {
		def, _ := json.Marshal(map[string]any{"procedure_code": "khai_sinh_v1", "name": "Đăng ký khai sinh", "slots": map[string]any{},
			"guidance": map[string]any{"summary": "", "checklist": []string{}}, "citations": c})
		return Contribution{Label: label, Definition: def}
	}
	agg, _ := Aggregate("khai_sinh_v1", "Đăng ký khai sinh", []Contribution{mk("batch:1", cites(0, 8)), mk("batch:2", cites(5, 13))}, nil)
	if tr := agg.Truncated[0]; len(tr.DroppedCitations) != 3 || tr.Dropped[0] != "Văn bản 10" {
		t.Fatalf("dropped citations = %+v", tr)
	}
	row := storeJobDraft(t, env, agg, nil)
	v := env.draft(t, row.ID)
	v, err := env.svc.AcknowledgeGap(ctx, ldUser, v.ID, AcknowledgeGapRequest{ExpectedUpdatedAt: &v.UpdatedAt, Field: "truncated:citations", Reason: "Đã xem 3 văn bản bị cắt."})
	if err != nil || issueCodes(v)["CONTENT_TRUNCATED"] != 0 {
		t.Fatalf("ack err=%v codes=%+v", err, issueCodes(v))
	}

	zero := 0
	v, err = env.svc.KeepCitation(ctx, ldUser, v.ID, KeepCitationRequest{ExpectedUpdatedAt: &v.UpdatedAt, Dropped: 1, Replace: &zero})
	if err != nil {
		t.Fatal(err)
	}
	var def aggDef
	_ = json.Unmarshal(v.Definition, &def)
	if len(def.Citations) != 10 || citationLabel(def.Citations[0]) != "Văn bản 11" {
		t.Fatalf("swap: first citation = %q (n=%d)", citationLabel(def.Citations[0]), len(def.Citations))
	}
	tr := v.Coverage.Truncated[0]
	if strings.Join(tr.Dropped, "|") != "Văn bản 10|Văn bản 12|Văn bản 0" {
		t.Fatalf("swapped-out citation must become dropped: %q", tr.Dropped)
	}
	if issueCodes(v)["CONTENT_TRUNCATED"] != 1 {
		t.Fatalf("a changed dropped list needs a fresh review: %+v", issueCodes(v))
	}
	for i := 0; i < 3; i++ {
		v, err = env.svc.KeepCitation(ctx, ldUser, v.ID, KeepCitationRequest{ExpectedUpdatedAt: &v.UpdatedAt, Dropped: 0})
		if err != nil {
			t.Fatal(err)
		}
	}
	_ = json.Unmarshal(v.Definition, &def)
	if len(def.Citations) != 13 || len(v.Coverage.Truncated) != 0 || issueCodes(v)["CONTENT_TRUNCATED"] != 0 {
		t.Fatalf("all kept: n=%d truncated=%+v codes=%+v", len(def.Citations), v.Coverage.Truncated, issueCodes(v))
	}
	if _, err := env.svc.KeepCitation(ctx, ldUser, v.ID, KeepCitationRequest{ExpectedUpdatedAt: &v.UpdatedAt, Dropped: 0}); err == nil {
		t.Fatal("nothing left to keep must fail")
	}
}

func citeRaw(docID, title string) json.RawMessage {
	raw, _ := json.Marshal(map[string]any{"doc_id": docID, "title": title, "source_type": "ai_draft_unapproved"})
	return raw
}

// The review basis of dropped citations follows source identity, not labels.
func TestCitationTruncationBasisUsesSourceIdentity(t *testing.T) {
	basis := func(cs ...json.RawMessage) string {
		t := Truncation{Field: "citations", DroppedCitations: cs}
		for _, c := range cs {
			t.Dropped = append(t.Dropped, citationLabel(c))
		}
		return truncationBasis(t)
	}
	a, b, c := citeRaw("doc-A", "Quyết định 163"), citeRaw("doc-B", "Quyết định 163"), citeRaw("doc-C", "Nghị định 18")
	if basis(a) == basis(b) {
		t.Fatal("same title, different document must give different bases")
	}
	if basis(a, c) == basis(b, c) {
		t.Fatal("a changed source must invalidate the review")
	}
	if basis(a) == basis(a, c) {
		t.Fatal("adding a citation must invalidate the review")
	}
	if basis(a, c, b) == basis(a, c) {
		t.Fatal("removing a citation must invalidate the review")
	}
	// The same sources written differently (key order, spacing, case, doc_id
	// vs document_id) are an unchanged set and keep the basis.
	a2 := json.RawMessage(`{"source_type":"ai_draft_unapproved","title":"  quyết định   163 ","doc_id":" DOC-A "}`)
	c2 := json.RawMessage(`{"document_id":"doc-C","title":"Nghị định 18","source_type":"ai_draft_unapproved"}`)
	if basis(a, c) != basis(a2, c2) {
		t.Fatal("an unchanged set of sources must keep its basis")
	}
	if basis(a, c) != basis(c, a) {
		t.Fatal("order does not matter for dropped citations")
	}
	if strings.Contains(basis(a), "Quyết định") || strings.Contains(basis(a), "doc-A") {
		t.Fatal("the basis must be a hash, not the citation text")
	}
}

func TestKeepCitationNeverDuplicates(t *testing.T) {
	env := newLDEnv(t)
	ctx := context.Background()
	mk := func(label string, c []any) Contribution {
		def, _ := json.Marshal(map[string]any{"procedure_code": "khai_sinh_v1", "name": "Đăng ký khai sinh", "slots": map[string]any{},
			"guidance": map[string]any{"summary": "", "checklist": []string{}}, "citations": c})
		return Contribution{Label: label, Definition: def}
	}
	cites := []any{}
	for i := 0; i < 10; i++ {
		cites = append(cites, map[string]any{"doc_id": fmt.Sprintf("d%d", i), "title": "Quyết định 163", "source_type": "ai_draft_unapproved"})
	}
	// Two extra sources share the title "Quyết định 163" with every kept one:
	// identity, not title, decides what is a duplicate.
	extra := []any{
		map[string]any{"doc_id": "d10", "title": "Quyết định 163", "source_type": "ai_draft_unapproved"},
		map[string]any{"doc_id": "d11", "title": "Quyết định 163", "source_type": "ai_draft_unapproved"},
	}
	agg, _ := Aggregate("khai_sinh_v1", "Đăng ký khai sinh", []Contribution{mk("batch:1", cites), mk("batch:2", extra)}, nil)
	if len(agg.Truncated) != 1 || len(agg.Truncated[0].DroppedCitations) != 2 {
		t.Fatalf("distinct sources with one title must not be merged: %+v", agg.Truncated)
	}
	row := storeJobDraft(t, env, agg, nil)
	v := env.draft(t, row.ID)

	// The Admin adds d10 by hand in the JSON, then presses "Giữ" on it.
	var obj map[string]any
	_ = json.Unmarshal(v.Definition, &obj)
	obj["citations"] = append(obj["citations"].([]any), extra[0])
	raw, _ := json.Marshal(obj)
	v, err := env.svc.Patch(ctx, ldUser, v.ID, PatchRequest{Definition: raw, ExpectedUpdatedAt: &v.UpdatedAt})
	if err != nil {
		t.Fatal(err)
	}
	zero := 0
	if _, err := env.svc.KeepCitation(ctx, ldUser, v.ID, KeepCitationRequest{ExpectedUpdatedAt: &v.UpdatedAt, Dropped: 0, Replace: &zero}); err == nil ||
		!strings.Contains(err.Error(), "already citation 11") {
		t.Fatalf("replacing with an already present source must be refused, got %v", err)
	}
	v, err = env.svc.KeepCitation(ctx, ldUser, v.ID, KeepCitationRequest{ExpectedUpdatedAt: &v.UpdatedAt, Dropped: 0})
	if err != nil {
		t.Fatal(err)
	}
	var def aggDef
	_ = json.Unmarshal(v.Definition, &def)
	if len(def.Citations) != 11 {
		t.Fatalf("an already kept source must not be added twice: %d", len(def.Citations))
	}
	if tr := v.Coverage.Truncated; len(tr) != 1 || len(tr[0].DroppedCitations) != 1 || citationIdentity(tr[0].DroppedCitations[0]) != citationIdentity(citeRaw("d11", "Quyết định 163")) {
		t.Fatalf("d10 must leave the dropped list, d11 stays: %+v", tr)
	}
	// Replacing keeps the count and moves the replaced source to the dropped list.
	v, err = env.svc.KeepCitation(ctx, ldUser, v.ID, KeepCitationRequest{ExpectedUpdatedAt: &v.UpdatedAt, Dropped: 0, Replace: &zero})
	if err != nil {
		t.Fatal(err)
	}
	_ = json.Unmarshal(v.Definition, &def)
	if len(def.Citations) != 11 || citationIdentity(v.Coverage.Truncated[0].DroppedCitations[0]) != citationIdentity(citeRaw("d0", "Quyết định 163")) {
		t.Fatalf("replace must not lose the replaced source: n=%d dropped=%s", len(def.Citations), v.Coverage.Truncated[0].DroppedCitations)
	}
}

// Citation contract at the draft layer: 10/11/15/20 are valid drafts, 21 is
// blocked by the validator and cannot be reached through KeepCitation.
func TestDraftCitationContractMatrix(t *testing.T) {
	for _, n := range []int{10, 11, 15, 20, 21} {
		cites := []any{}
		for i := 0; i < n; i++ {
			cites = append(cites, map[string]any{"doc_id": fmt.Sprintf("d%d", i), "title": fmt.Sprintf("Văn bản %d", i), "source_type": "ai_draft_unapproved"})
		}
		raw, _ := json.Marshal(map[string]any{
			"procedure_code": "khai_sinh_v1", "name": "Đăng ký khai sinh", "intent_examples": []string{"khai sinh"},
			"slots":          map[string]any{"hinh_thuc_nop": map[string]any{"type": "string", "question": "Hình thức nộp?"}},
			"required_slots": []string{}, "conditional_slots": []any{},
			"guidance":  map[string]any{"summary": "s", "checklist": []string{"a"}, "where_to_submit": "UBND"},
			"citations": cites,
		})
		val := decision.ValidateDraftDefinition(raw, "khai_sinh_v1")
		blocked := false
		for _, is := range val.Issues {
			blocked = blocked || is.Code == "TOO_MANY_CITATIONS"
		}
		if blocked != (n > decision.MaxDraftCitations) {
			t.Fatalf("n=%d: TOO_MANY_CITATIONS=%v", n, blocked)
		}
		def, err := decision.ParseDefinition(raw)
		if err != nil || len(def.Citations) != n {
			t.Fatalf("n=%d: parse err=%v count=%d", n, err, len(def.Citations))
		}
		// Persistence round trip keeps every citation.
		payload, _ := json.Marshal(storedPayloadFromView(DraftView{Definition: raw}))
		back, err := viewFromRow(repository.ProcedureDraft{DraftDefinition: payload})
		if err != nil {
			t.Fatal(err)
		}
		var obj aggDef
		_ = json.Unmarshal(back.Definition, &obj)
		if len(obj.Citations) != n {
			t.Fatalf("n=%d: stored %d citations", n, len(obj.Citations))
		}
		view := DraftView{Validation: val, Definition: raw}
		if gateErr := approvalGate(view); n > decision.MaxDraftCitations && (gateErr == nil || !errors.Is(gateErr, ErrApprovalBlocked)) {
			t.Fatalf("n=%d: approval must be blocked, got %v", n, gateErr)
		}
	}
}

func TestCitationIdentityNormalisesSourceID(t *testing.T) {
	byDocID := json.RawMessage(`{"doc_id":"8939d98f","title":"Quyết định 163","source_type":"ai_draft_unapproved"}`)
	byDocumentID := json.RawMessage(`{"document_id":"8939D98F","title":"Quyết định 163","source_type":"ai_draft_unapproved"}`)
	other := json.RawMessage(`{"document_id":"1111aaaa","title":"Quyết định 163","source_type":"ai_draft_unapproved"}`)
	if citationIdentity(byDocID) != citationIdentity(byDocumentID) {
		t.Fatal("doc_id and document_id of the same document must be one source")
	}
	if citationIdentity(byDocID) == citationIdentity(other) {
		t.Fatal("different documents with the same title must stay different")
	}
	// A free-text ID is not a document reference: number and title still count.
	numbered := json.RawMessage(`{"doc_id":"8939d98f","title":"Quyết định 163","document_number":"163/QĐ-BTP","source_type":"ai_draft_unapproved"}`)
	if citationIdentity(byDocID) == citationIdentity(numbered) {
		t.Fatal("a document number is part of a free-text identity")
	}
}

// Real drafts cite the source PDF by its UUID under several titles; that is
// one document, so one citation.
func TestCitationIdentityByDocumentUUID(t *testing.T) {
	const doc = "8939d98f-ce2c-487f-9f28-a53510c29271"
	titles := []string{
		"Quyết định 163",
		"Quyết định số 163/QĐ-BTP ngày 16/02/2026 của Bộ Tư pháp",
		"Quyết định số 163/QĐ-BTP ngày 16/02/2026 của Bộ trưởng Bộ Tư pháp",
	}
	first := citationIdentity(citeRaw(doc, titles[0]))
	for _, title := range titles[1:] {
		if citationIdentity(citeRaw(doc, title)) != first {
			t.Fatalf("same document UUID, title %q must be the same source", title)
		}
	}
	upper := json.RawMessage(`{"document_id":"8939D98F-CE2C-487F-9F28-A53510C29271","title":"x","document_number":"163","source_type":"ai_draft_unapproved"}`)
	if citationIdentity(upper) != first {
		t.Fatal("document_id/doc_id, case and number must not split a UUID source")
	}
	hexForm := json.RawMessage(`{"doc_id":"8939d98fce2c487f9f28a53510c29271","document_id":"8939D98F-CE2C-487F-9F28-A53510C29271","title":"y","source_type":"ai_draft_unapproved"}`)
	if citationIdentity(hexForm) != first {
		t.Fatal("two spellings of one UUID are one source, not a conflict")
	}
	if citationIdentity(citeRaw("1111aaaa-ce2c-487f-9f28-a53510c29271", titles[0])) == first {
		t.Fatal("another document UUID is another source")
	}

	// Conflicting IDs: changing either one changes the identity.
	conflict := func(documentID string) json.RawMessage {
		raw, _ := json.Marshal(map[string]any{"doc_id": doc, "document_id": documentID, "title": titles[0], "source_type": "ai_draft_unapproved"})
		return raw
	}
	b, c := conflict("document-b"), conflict("document-c")
	if citationIdentity(b) == citationIdentity(c) || citationIdentity(b) == first {
		t.Fatal("a conflicting document_id must never be ignored")
	}

	var cs []Contribution
	for i, title := range titles {
		def, _ := json.Marshal(map[string]any{"procedure_code": "khai_sinh_v1", "name": "Đăng ký khai sinh", "slots": map[string]any{},
			"guidance":  map[string]any{"summary": ""},
			"citations": []any{map[string]any{"doc_id": doc, "title": title, "source_type": "ai_draft_unapproved"}}})
		cs = append(cs, Contribution{Label: fmt.Sprintf("batch:%d", i+1), Definition: def})
	}
	res, err := Aggregate("khai_sinh_v1", "Đăng ký khai sinh", cs, nil)
	if err != nil {
		t.Fatal(err)
	}
	var out aggDef
	_ = json.Unmarshal(res.Definition, &out)
	if len(out.Citations) != 1 {
		t.Fatalf("one document cited by 3 batches must be 1 citation, got %d", len(out.Citations))
	}
}

// One document described differently by different batches: the merge keeps
// one citation, opens a conflict with every description and does not pick one.
func TestCitationDescriptionConflictNeedsAdminChoice(t *testing.T) {
	env := newLDEnv(t)
	ctx := context.Background()
	const doc = "8939d98f-ce2c-487f-9f28-a53510c29271"
	mk := func(label string, cites ...map[string]any) Contribution {
		list := []any{}
		for _, c := range cites {
			c["source_type"] = "ai_draft_unapproved"
			list = append(list, c)
		}
		def, _ := json.Marshal(map[string]any{"procedure_code": "khai_sinh_v1", "name": "Đăng ký khai sinh", "slots": map[string]any{},
			"guidance": map[string]any{"summary": ""}, "citations": list})
		return Contribution{Label: label, PageStart: 9, PageEnd: 15, Definition: def}
	}
	short := map[string]any{"doc_id": doc, "title": "Quyết định 163"}
	full := map[string]any{"doc_id": doc, "title": "Quyết định số 163/QĐ-BTP ngày 16/02/2026 của Bộ Tư pháp", "document_number": "163/QĐ-BTP"}
	sameAsShort := map[string]any{"doc_id": strings.ToUpper(doc), "title": "  quyết định 163 "}
	other := map[string]any{"doc_id": "1111aaaa-ce2c-487f-9f28-a53510c29271", "title": "Luật Hộ tịch"}
	agg, err := Aggregate("khai_sinh_v1", "Đăng ký khai sinh", []Contribution{
		mk("batch:1", short, other), mk("batch:2", full), mk("batch:3", sameAsShort, other),
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	var def aggDef
	_ = json.Unmarshal(agg.Definition, &def)
	if len(def.Citations) != 2 {
		t.Fatalf("citations = %d, want 2 (one per document)", len(def.Citations))
	}
	if len(agg.Conflicts) != 1 {
		t.Fatalf("conflicts = %+v, want one for the doubly described document", agg.Conflicts)
	}
	cf := agg.Conflicts[0]
	if cf.Field != "citations" || cf.Status != ConflictOpen || len(cf.Candidates) != 2 ||
		cf.Candidates[0].Contributor != "batch:1" || cf.Candidates[1].Contributor != "batch:2" ||
		!strings.Contains(cf.Candidates[1].Value, "số 163/QĐ-BTP") {
		t.Fatalf("conflict = %+v", cf)
	}

	row := storeJobDraft(t, env, agg, nil)
	v := env.draft(t, row.ID)
	if issueCodes(v)["SOURCE_CONFLICT"] != 1 || approvalGate(v) == nil {
		t.Fatalf("an open description conflict must block approval: %+v", issueCodes(v))
	}
	one := 1
	out, err := env.svc.ResolveConflict(ctx, ldUser, v.ID, ResolveConflictRequest{ExpectedUpdatedAt: &v.UpdatedAt, ConflictID: cf.ID, Choice: &one})
	if err != nil {
		t.Fatal(err)
	}
	var got struct {
		Citations []decision.Citation `json:"citations"`
	}
	_ = json.Unmarshal(out.Definition, &got)
	if len(got.Citations) != 2 || got.Citations[0].Title != full["title"] || got.Citations[0].DocumentNumber != "163/QĐ-BTP" || got.Citations[1].Title != "Luật Hộ tịch" {
		t.Fatalf("chosen description must replace the citation in place: %+v", got.Citations)
	}
	if out.Conflicts[0].Status != ConflictResolved || issueCodes(out)["SOURCE_CONFLICT"] != 0 {
		t.Fatalf("conflict = %+v issues = %+v", out.Conflicts[0], issueCodes(out))
	}

	// The document was removed from the JSON: only "keep current" is possible.
	row2 := storeJobDraft(t, env, agg, nil)
	v2 := env.draft(t, row2.ID)
	var obj map[string]any
	_ = json.Unmarshal(v2.Definition, &obj)
	obj["citations"] = []any{map[string]any{"doc_id": other["doc_id"], "title": "Luật Hộ tịch", "source_type": "ai_draft_unapproved"}}
	raw, _ := json.Marshal(obj)
	v2, err = env.svc.Patch(ctx, ldUser, v2.ID, PatchRequest{Definition: raw, ExpectedUpdatedAt: &v2.UpdatedAt})
	if err != nil {
		t.Fatal(err)
	}
	zero, keep := 0, -1
	if _, err := env.svc.ResolveConflict(ctx, ldUser, v2.ID, ResolveConflictRequest{ExpectedUpdatedAt: &v2.UpdatedAt, ConflictID: cf.ID, Choice: &zero}); err == nil ||
		!strings.Contains(err.Error(), "no longer cited") {
		t.Fatalf("choosing a description for an uncited document must fail, got %v", err)
	}
	if _, err := env.svc.ResolveConflict(ctx, ldUser, v2.ID, ResolveConflictRequest{ExpectedUpdatedAt: &v2.UpdatedAt, ConflictID: cf.ID, Choice: &keep}); err != nil {
		t.Fatal(err)
	}
}

// SUPPLEMENT: an unresolved description conflict of the base is carried into
// the new merge, so the base's other description is not lost.
func TestCitationDescriptionConflictCarriedFromBase(t *testing.T) {
	const doc = "8939d98f-ce2c-487f-9f28-a53510c29271"
	cite := func(title string) json.RawMessage {
		raw, _ := json.Marshal(map[string]any{"doc_id": doc, "title": title, "source_type": "ai_draft_unapproved"})
		return raw
	}
	def, _ := json.Marshal(map[string]any{"procedure_code": "khai_sinh_v1", "name": "Đăng ký khai sinh", "slots": map[string]any{},
		"guidance": map[string]any{"summary": ""}, "citations": []any{json.RawMessage(cite("Quyết định 163"))}})
	id := citationConflictID(citationIdentity(cite("x")))
	base := Contribution{Label: "base", Definition: def, OpenConflicts: []DraftConflict{{ID: id, Field: "citations", Status: ConflictOpen,
		Candidates: []ConflictCandidate{{Value: "Quyết định 163", Contributor: "batch:1", Citation: cite("Quyết định 163")},
			{Value: "Quyết định số 163/QĐ-BTP", Contributor: "batch:2", Citation: cite("Quyết định số 163/QĐ-BTP")}}}}}
	batch := Contribution{Label: "batch:1", Definition: def}
	agg, err := Aggregate("khai_sinh_v1", "Đăng ký khai sinh", []Contribution{base, batch}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(agg.Conflicts) != 1 || agg.Conflicts[0].ID != id || len(agg.Conflicts[0].Candidates) != 2 {
		t.Fatalf("base conflict must survive the merge: %+v", agg.Conflicts)
	}
}

// citeDocs: n citations of distinct document UUIDs (doc 0..n-1).
func citeDocs(from, to int, title func(i int) string) []any {
	out := []any{}
	for i := from; i < to; i++ {
		out = append(out, map[string]any{"doc_id": fmt.Sprintf("00000000-0000-4000-8000-%012d", i), "title": title(i), "source_type": "ai_draft_unapproved"})
	}
	return out
}

func citeContribution(label string, cites []any) Contribution {
	def, _ := json.Marshal(map[string]any{"procedure_code": "khai_sinh_v1", "name": "Đăng ký khai sinh", "slots": map[string]any{},
		"guidance": map[string]any{"summary": ""}, "citations": cites})
	return Contribution{Label: label, PageStart: 9, PageEnd: 21, Definition: def}
}

// Document 11 is merge-dropped and described two ways: its conflict is kept
// (inactive, not blocking) and becomes blocking once the Admin cites it again.
func TestDroppedCitationKeepsDescriptionConflict(t *testing.T) {
	env := newLDEnv(t)
	ctx := context.Background()
	plain := func(i int) string { return fmt.Sprintf("Văn bản %d", i) }
	agg, err := Aggregate("khai_sinh_v1", "Đăng ký khai sinh", []Contribution{
		citeContribution("batch:1", citeDocs(0, 11, plain)),
		citeContribution("batch:2", citeDocs(10, 11, func(int) string { return "Quyết định số 163/QĐ-BTP" })),
		citeContribution("batch:3", citeDocs(10, 11, plain)),
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(agg.Conflicts) != 1 || agg.Conflicts[0].Field != "citations" || len(agg.Conflicts[0].Candidates) != 2 {
		t.Fatalf("the dropped document's conflict must be kept: %+v", agg.Conflicts)
	}
	if got := agg.Conflicts[0].Candidates[0]; got.Contributor != "batch:1" || len(got.AlsoFrom) != 1 || got.AlsoFrom[0] != "batch:3" {
		t.Fatalf("every batch proposing a description must be recorded: %+v", got)
	}
	row := storeJobDraft(t, env, agg, nil)
	v := env.draft(t, row.ID)
	if !v.Conflicts[0].Inactive || issueCodes(v)["SOURCE_CONFLICT"] != 0 {
		t.Fatalf("an uncited document's conflict must not block: %+v %+v", v.Conflicts[0], issueCodes(v))
	}

	// The Admin keeps document 10 in place of citation 1.
	zero := 0
	v, err = env.svc.KeepCitation(ctx, ldUser, v.ID, KeepCitationRequest{ExpectedUpdatedAt: &v.UpdatedAt, Dropped: 0, Replace: &zero})
	if err != nil {
		t.Fatal(err)
	}
	if v.Conflicts[0].Inactive || v.Conflicts[0].Status != ConflictOpen || issueCodes(v)["SOURCE_CONFLICT"] != 1 || approvalGate(v) == nil {
		t.Fatalf("a kept document must bring its conflict back: %+v %+v", v.Conflicts[0], issueCodes(v))
	}
	one := 1
	v, err = env.svc.ResolveConflict(ctx, ldUser, v.ID, ResolveConflictRequest{ExpectedUpdatedAt: &v.UpdatedAt, ConflictID: v.Conflicts[0].ID, Choice: &one})
	if err != nil {
		t.Fatal(err)
	}
	var def struct {
		Citations []decision.Citation `json:"citations"`
	}
	_ = json.Unmarshal(v.Definition, &def)
	if def.Citations[0].Title != "Quyết định số 163/QĐ-BTP" || issueCodes(v)["SOURCE_CONFLICT"] != 0 {
		t.Fatalf("citations = %+v issues = %+v", def.Citations, issueCodes(v))
	}
}

// A decision is pinned to the value it produced: editing that value in the
// JSON reopens the conflict; editing anything else does not.
func TestEditedResolvedValueReopensConflict(t *testing.T) {
	env := newLDEnv(t)
	ctx := context.Background()
	const doc = "8939d98f-ce2c-487f-9f28-a53510c29271"
	cite := func(title string) []any {
		return []any{map[string]any{"doc_id": doc, "title": title, "source_type": "ai_draft_unapproved"}}
	}
	agg, err := Aggregate("khai_sinh_v1", "Đăng ký khai sinh", []Contribution{
		citeContribution("batch:1", cite("Quyết định 163")), citeContribution("batch:2", cite("Quyết định số 163/QĐ-BTP")),
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	row := storeJobDraft(t, env, agg, nil)
	v := env.draft(t, row.ID)
	one := 1
	v, err = env.svc.ResolveConflict(ctx, ldUser, v.ID, ResolveConflictRequest{ExpectedUpdatedAt: &v.UpdatedAt, ConflictID: v.Conflicts[0].ID, Choice: &one})
	if err != nil || v.Conflicts[0].Status != ConflictResolved || v.Conflicts[0].ResolvedBasis == "" {
		t.Fatalf("resolve: %+v err=%v", v.Conflicts, err)
	}
	patch := func(edit func(obj map[string]any)) DraftView {
		var obj map[string]any
		_ = json.Unmarshal(v.Definition, &obj)
		edit(obj)
		raw, _ := json.Marshal(obj)
		out, err := env.svc.Patch(ctx, ldUser, v.ID, PatchRequest{Definition: raw, ExpectedUpdatedAt: &v.UpdatedAt})
		if err != nil {
			t.Fatal(err)
		}
		return out
	}
	// Unrelated edit, and the same title re-spaced: the decision stands.
	v = patch(func(obj map[string]any) {
		obj["intent_examples"] = []string{"làm giấy khai sinh"}
		obj["citations"].([]any)[0].(map[string]any)["title"] = "  quyết định số 163/QĐ-BTP "
	})
	if v.Conflicts[0].Status != ConflictResolved || v.Conflicts[0].Reopened {
		t.Fatalf("an edit that keeps the chosen value must not reopen: %+v", v.Conflicts[0])
	}
	// The chosen description itself is changed: decide again.
	v = patch(func(obj map[string]any) {
		obj["citations"].([]any)[0].(map[string]any)["title"] = "Quyết định 999"
	})
	c := v.Conflicts[0]
	if c.Status != ConflictOpen || !c.Reopened || c.Chosen != nil || c.ResolvedBy != nil || issueCodes(v)["SOURCE_CONFLICT"] != 1 || approvalGate(v) == nil {
		t.Fatalf("an edited decision must reopen and block: %+v %+v", c, issueCodes(v))
	}
	keep := -1
	v, err = env.svc.ResolveConflict(ctx, ldUser, v.ID, ResolveConflictRequest{ExpectedUpdatedAt: &v.UpdatedAt, ConflictID: c.ID, Choice: &keep})
	if err != nil || v.Conflicts[0].Status != ConflictResolved || v.Conflicts[0].Reopened {
		t.Fatalf("keeping the edited value must resolve again: %+v err=%v", v.Conflicts[0], err)
	}
}

// The same pinning applies to legal guidance fields.
func TestEditedGuidanceDecisionReopens(t *testing.T) {
	env := newLDEnv(t)
	ctx := context.Background()
	job := env.createJob(t, DraftPlanRequest{DocumentID: ldDoc})
	env.runner.Drain(ctx)
	got, _ := env.svc.GetDraftJob(ctx, job.ID)
	v := env.draft(t, *got.DraftID)
	one := 1
	v, err := env.svc.ResolveConflict(ctx, ldUser, v.ID, ResolveConflictRequest{ExpectedUpdatedAt: &v.UpdatedAt, ConflictID: "guidance:fee_hint", Choice: &one})
	if err != nil {
		t.Fatal(err)
	}
	var obj map[string]any
	_ = json.Unmarshal(v.Definition, &obj)
	obj["guidance"].(map[string]any)["fee_hint"] = "Miễn lệ phí"
	raw, _ := json.Marshal(obj)
	v, err = env.svc.Patch(ctx, ldUser, v.ID, PatchRequest{Definition: raw, ExpectedUpdatedAt: &v.UpdatedAt})
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range v.Conflicts {
		if c.ID == "guidance:fee_hint" && (c.Status != ConflictOpen || !c.Reopened) {
			t.Fatalf("fee decision must reopen after the fee was edited: %+v", c)
		}
	}
}

// Punctuation in a legal number is meaningful: swapping "/" and "-" after a
// decision reopens it, while case and spacing do not.
func TestDocumentNumberPunctuationReopensDecision(t *testing.T) {
	env := newLDEnv(t)
	ctx := context.Background()
	const doc = "8939d98f-ce2c-487f-9f28-a53510c29271"
	cite := func(number string) []any {
		return []any{map[string]any{"doc_id": doc, "title": "Quyết định 163", "document_number": number, "source_type": "ai_draft_unapproved"}}
	}
	raw := func(number string) json.RawMessage {
		b, _ := json.Marshal(cite(number)[0])
		return b
	}
	if citationMeta(raw("163/QĐ-BTP")) == citationMeta(raw("163-QĐ/BTP")) {
		t.Fatal("different punctuation in a document number is a different value")
	}
	if citationMeta(raw("163/QĐ-BTP")) != citationMeta(raw(" 163/qđ-btp ")) {
		t.Fatal("case and spacing are not a different value")
	}
	agg, err := Aggregate("khai_sinh_v1", "Đăng ký khai sinh", []Contribution{
		citeContribution("batch:1", cite("163/QĐ-BTP")), citeContribution("batch:2", cite("163-QĐ/BTP")),
	}, nil)
	if err != nil || len(agg.Conflicts) != 1 {
		t.Fatalf("two punctuations are two candidates: %+v err=%v", agg.Conflicts, err)
	}
	row := storeJobDraft(t, env, agg, nil)
	v := env.draft(t, row.ID)
	zero := 0
	if v, err = env.svc.ResolveConflict(ctx, ldUser, v.ID, ResolveConflictRequest{ExpectedUpdatedAt: &v.UpdatedAt, ConflictID: v.Conflicts[0].ID, Choice: &zero}); err != nil {
		t.Fatal(err)
	}
	var obj map[string]any
	_ = json.Unmarshal(v.Definition, &obj)
	obj["citations"].([]any)[0].(map[string]any)["document_number"] = "163-QĐ/BTP"
	b, _ := json.Marshal(obj)
	v, err = env.svc.Patch(ctx, ldUser, v.ID, PatchRequest{Definition: b, ExpectedUpdatedAt: &v.UpdatedAt})
	if err != nil {
		t.Fatal(err)
	}
	if v.Conflicts[0].Status != ConflictOpen || !v.Conflicts[0].Reopened {
		t.Fatalf("a changed separator must reopen the decision: %+v", v.Conflicts[0])
	}
}

// A decision stored without a basis (before decisions were pinned) cannot be
// checked: an editable draft reads it as reopened and approval is blocked.
// Published drafts are left as they are.
func TestLegacyResolvedConflictFailsClosed(t *testing.T) {
	env := newLDEnv(t)
	ctx := context.Background()
	job := env.createJob(t, DraftPlanRequest{DocumentID: ldDoc})
	env.runner.Drain(ctx)
	got, _ := env.svc.GetDraftJob(ctx, job.ID)
	v := env.draft(t, *got.DraftID)
	one := 1
	v, err := env.svc.ResolveConflict(ctx, ldUser, v.ID, ResolveConflictRequest{ExpectedUpdatedAt: &v.UpdatedAt, ConflictID: "guidance:fee_hint", Choice: &one})
	if err != nil {
		t.Fatal(err)
	}
	if len(v.Provenance) == 0 {
		t.Fatal("test needs provenance from the decision")
	}
	// Simulate a fixes9 row: RESOLVED without resolved_basis.
	stored := env.drafts.rows[v.ID]
	var payload map[string]any
	_ = json.Unmarshal(stored.DraftDefinition, &payload)
	for _, c := range payload["conflicts"].([]any) {
		delete(c.(map[string]any), "resolved_basis")
	}
	stored.DraftDefinition, _ = json.Marshal(payload)
	env.drafts.rows[v.ID] = stored

	legacy, err := env.svc.Get(ctx, v.ID)
	if err != nil {
		t.Fatal(err)
	}
	var fee *DraftConflict
	for i := range legacy.Conflicts {
		if legacy.Conflicts[i].ID == "guidance:fee_hint" {
			fee = &legacy.Conflicts[i]
		}
	}
	if fee == nil || fee.Status != ConflictOpen || !fee.Reopened || fee.Chosen != nil {
		t.Fatalf("a basis-less decision must read as reopened: %+v", fee)
	}
	for _, p := range legacy.Provenance {
		if p.ProposalID == "guidance:fee_hint" {
			t.Fatalf("a reopened decision must not keep its provenance: %+v", p)
		}
	}
	if err := approvalGate(legacy); err == nil || !errors.Is(err, ErrApprovalBlocked) {
		t.Fatalf("approval must be blocked, got %v", err)
	}
	// Choosing again pins the decision.
	again, err := env.svc.ResolveConflict(ctx, ldUser, v.ID, ResolveConflictRequest{ExpectedUpdatedAt: &legacy.UpdatedAt, ConflictID: "guidance:fee_hint", Choice: &one})
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range again.Conflicts {
		if c.ID == "guidance:fee_hint" && (c.Status != ConflictResolved || c.ResolvedBasis == "" || c.Reopened) {
			t.Fatalf("re-decided conflict = %+v", c)
		}
	}

	// A published draft is history: not rewritten on read.
	stored.Status = repository.DraftStatusPublished
	env.drafts.rows[v.ID] = stored
	pub, err := env.svc.Get(ctx, v.ID)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range pub.Conflicts {
		if c.ID == "guidance:fee_hint" && c.Status != ConflictResolved {
			t.Fatalf("published draft conflict changed on read: %+v", c)
		}
	}
}

// When an edit reopens a decision, the provenance that decision set is removed.
func TestReopenedDecisionDropsProvenance(t *testing.T) {
	env := newLDEnv(t)
	ctx := context.Background()
	job := env.createJob(t, DraftPlanRequest{DocumentID: ldDoc})
	env.runner.Drain(ctx)
	got, _ := env.svc.GetDraftJob(ctx, job.ID)
	v := env.draft(t, *got.DraftID)
	one := 1
	v, err := env.svc.ResolveConflict(ctx, ldUser, v.ID, ResolveConflictRequest{ExpectedUpdatedAt: &v.UpdatedAt, ConflictID: "guidance:fee_hint", Choice: &one})
	if err != nil {
		t.Fatal(err)
	}
	has := func(view DraftView) bool {
		for _, p := range view.Provenance {
			if p.ProposalID == "guidance:fee_hint" {
				return true
			}
		}
		return false
	}
	if !has(v) {
		t.Fatal("the decision must set provenance")
	}
	var obj map[string]any
	_ = json.Unmarshal(v.Definition, &obj)
	obj["guidance"].(map[string]any)["fee_hint"] = "Miễn lệ phí"
	b, _ := json.Marshal(obj)
	v, err = env.svc.Patch(ctx, ldUser, v.ID, PatchRequest{Definition: b, ExpectedUpdatedAt: &v.UpdatedAt})
	if err != nil {
		t.Fatal(err)
	}
	if has(v) {
		t.Fatal("a reopened decision must not keep showing its old sources")
	}
}

// Read → reopened → issues match → Approve blocked → Resolve → consistent.
func TestReopenedOnReadKeepsValidationConsistent(t *testing.T) {
	env := newLDEnv(t)
	ctx := context.Background()
	job := env.createJob(t, DraftPlanRequest{DocumentID: ldDoc})
	env.runner.Drain(ctx)
	got, _ := env.svc.GetDraftJob(ctx, job.ID)
	v := env.draft(t, *got.DraftID)
	open := 0
	for _, c := range v.Conflicts {
		if c.Status != ConflictResolved {
			open++
		}
	}
	// Resolve every conflict, then strip the bases (a row from before pinning).
	for _, c := range v.Conflicts {
		zero := 0
		var err error
		if v, err = env.svc.ResolveConflict(ctx, ldUser, v.ID, ResolveConflictRequest{ExpectedUpdatedAt: &v.UpdatedAt, ConflictID: c.ID, Choice: &zero}); err != nil {
			t.Fatal(err)
		}
	}
	if issueCodes(v)["SOURCE_CONFLICT"] != 0 || open == 0 {
		t.Fatalf("setup: open=%d issues=%+v", open, issueCodes(v))
	}
	stored := env.drafts.rows[v.ID]
	var payload map[string]any
	_ = json.Unmarshal(stored.DraftDefinition, &payload)
	for _, c := range payload["conflicts"].([]any) {
		delete(c.(map[string]any), "resolved_basis")
	}
	stored.DraftDefinition, _ = json.Marshal(payload)
	env.drafts.rows[v.ID] = stored

	read, err := env.svc.Get(ctx, v.ID)
	if err != nil {
		t.Fatal(err)
	}
	reopened := 0
	for _, c := range read.Conflicts {
		if c.Status == ConflictOpen && c.Reopened {
			reopened++
		}
	}
	if reopened != len(read.Conflicts) || issueCodes(read)["SOURCE_CONFLICT"] != reopened {
		t.Fatalf("a read must report one SOURCE_CONFLICT per reopened conflict: reopened=%d issues=%+v", reopened, issueCodes(read))
	}
	if _, err := env.svc.Approve(ctx, ldUser, read.ID, ApproveRequest{Confirm: true, ExpectedUpdatedAt: &read.UpdatedAt}); err == nil || !errors.Is(err, ErrApprovalBlocked) {
		t.Fatalf("approve must be blocked, got %v", err)
	}
	cur := read
	for _, c := range read.Conflicts {
		zero := 0
		if cur, err = env.svc.ResolveConflict(ctx, ldUser, cur.ID, ResolveConflictRequest{ExpectedUpdatedAt: &cur.UpdatedAt, ConflictID: c.ID, Choice: &zero}); err != nil {
			t.Fatal(err)
		}
	}
	again, _ := env.svc.Get(ctx, cur.ID)
	for _, c := range again.Conflicts {
		if c.Status != ConflictResolved || c.ResolvedBasis == "" || c.Reopened {
			t.Fatalf("re-decided conflict = %+v", c)
		}
	}
	if issueCodes(again)["SOURCE_CONFLICT"] != 0 || issueCodes(cur)["SOURCE_CONFLICT"] != 0 {
		t.Fatalf("no SOURCE_CONFLICT may remain: read=%+v saved=%+v", issueCodes(again), issueCodes(cur))
	}
}

// A published Definition can be corrected without drafting again: the copy is
// an editable DRAFT with the same content and reviews, linked to its parent,
// without the approval; asking again returns the same open copy.
func TestCloneDraftFromPublished(t *testing.T) {
	env := newLDEnv(t)
	ctx := context.Background()
	job := env.createJob(t, DraftPlanRequest{DocumentID: ldDoc})
	env.runner.Drain(ctx)
	got, _ := env.svc.GetDraftJob(ctx, job.ID)
	src := env.draft(t, *got.DraftID)
	if _, err := env.svc.CloneDraft(ctx, ldUser, src.ID); err == nil {
		t.Fatal("a DRAFT is edited directly, not copied")
	}
	row := env.drafts.rows[src.ID]
	row.Status = repository.DraftStatusPublished
	env.drafts.rows[src.ID] = row

	cp, err := env.svc.CloneDraft(ctx, ldUser, src.ID)
	if err != nil {
		t.Fatal(err)
	}
	if cp.ID == src.ID || cp.Status != repository.DraftStatusDraft || cp.ParentDraftID == nil || *cp.ParentDraftID != src.ID {
		t.Fatalf("copy = id %s status %s parent %v", cp.ID, cp.Status, cp.ParentDraftID)
	}
	if string(cp.Definition) != string(src.Definition) || len(cp.Evidence) != len(src.Evidence) || len(cp.Conflicts) != len(src.Conflicts) {
		t.Fatal("the copy must keep the Definition, evidence and conflicts")
	}
	if cp.ApprovedBy != nil || cp.PublishedVersionID != nil || cp.JobID != nil || cp.Validation.SemanticApproved {
		t.Fatalf("approval/publication must not be copied: %+v", cp)
	}
	again, err := env.svc.CloneDraft(ctx, ldUser, src.ID)
	if err != nil || again.ID != cp.ID {
		t.Fatalf("a second copy request must return the open copy: %v %v", again.ID, err)
	}
}

// A proposal the document does not state can be kept on an Admin's reason:
// it no longer blocks approval, is reported as a decision, follows a move,
// and is lost when its content is edited.
func TestAdminDecisionWithoutSource(t *testing.T) {
	env := newLDEnv(t)
	ctx := context.Background()
	job := env.createJob(t, DraftPlanRequest{DocumentID: ldDoc})
	env.runner.Drain(ctx)
	got, _ := env.svc.GetDraftJob(ctx, job.ID)
	v := env.draft(t, *got.DraftID)

	// The Admin adds a checklist item the source does not state.
	var obj map[string]any
	_ = json.Unmarshal(v.Definition, &obj)
	g := obj["guidance"].(map[string]any)
	g["checklist"] = append([]any{"Bản sao sổ hộ khẩu (quy định của xã)"}, g["checklist"].([]any)...)
	raw, _ := json.Marshal(obj)
	v, err := env.svc.Patch(ctx, ldUser, v.ID, PatchRequest{Definition: raw, ExpectedUpdatedAt: &v.UpdatedAt})
	if err != nil {
		t.Fatal(err)
	}
	missing := func(view DraftView) bool {
		for _, is := range view.Validation.Issues {
			if is.Code == "MISSING_EVIDENCE" && is.ProposalID == "checklist:0" {
				return true
			}
		}
		return false
	}
	if !missing(v) {
		t.Fatal("setup: the added item has no source")
	}
	if _, err := env.svc.DecideWithoutSource(ctx, ldUser, v.ID, AdminDecisionRequest{ExpectedUpdatedAt: &v.UpdatedAt, ProposalID: "checklist:0", Reason: "ngắn"}); err == nil {
		t.Fatal("a reason of at least 10 characters is required")
	}
	if _, err := env.svc.DecideWithoutSource(ctx, ldUser, v.ID, AdminDecisionRequest{ExpectedUpdatedAt: &v.UpdatedAt, ProposalID: "checklist:1", Reason: "văn bản không nêu nhưng xã yêu cầu"}); err == nil {
		t.Fatal("a proposal with a valid quotation must be accepted, not decided")
	}
	v, err = env.svc.DecideWithoutSource(ctx, ldUser, v.ID, AdminDecisionRequest{ExpectedUpdatedAt: &v.UpdatedAt, ProposalID: "checklist:0", Reason: "Quy định riêng của UBND xã, văn bản không nêu"})
	if err != nil {
		t.Fatal(err)
	}
	if missing(v) || issueCodes(v)["ADMIN_DECISION_NO_SOURCE"] != 1 || len(v.AdminDecisions) != 1 || v.AdminDecisions[0].By != ldUser {
		t.Fatalf("decision not applied: %+v %+v", issueCodes(v), v.AdminDecisions)
	}
	for _, is := range v.Validation.Issues {
		if is.ProposalID == "checklist:0" && is.Severity == "error" {
			t.Fatalf("a decided proposal must not block: %+v", is)
		}
	}

	// Another item is inserted before it: the decision follows the content.
	_ = json.Unmarshal(v.Definition, &obj)
	g = obj["guidance"].(map[string]any)
	g["checklist"] = append([]any{"Mục mới khác"}, g["checklist"].([]any)...)
	raw, _ = json.Marshal(obj)
	v, err = env.svc.Patch(ctx, ldUser, v.ID, PatchRequest{Definition: raw, ExpectedUpdatedAt: &v.UpdatedAt})
	if err != nil {
		t.Fatal(err)
	}
	if len(v.AdminDecisions) != 1 || v.AdminDecisions[0].ProposalID != "checklist:1" {
		t.Fatalf("decision must follow the moved item: %+v", v.AdminDecisions)
	}

	// Editing the decided text drops the decision.
	_ = json.Unmarshal(v.Definition, &obj)
	g = obj["guidance"].(map[string]any)
	g["checklist"].([]any)[1] = "Bản sao sổ hộ khẩu (đã sửa)"
	raw, _ = json.Marshal(obj)
	v, err = env.svc.Patch(ctx, ldUser, v.ID, PatchRequest{Definition: raw, ExpectedUpdatedAt: &v.UpdatedAt})
	if err != nil {
		t.Fatal(err)
	}
	if len(v.AdminDecisions) != 0 {
		t.Fatalf("an edited proposal needs a new decision: %+v", v.AdminDecisions)
	}
}
