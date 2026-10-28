package drafting

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgconn"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/decision"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/repository"
)

// Long-document drafting: one job drafts ONE procedure section in several model
// calls ("batches"), stores each batch result, then merges them (Aggregate) into
// a single new draft. The whole document is only scanned for section headings;
// it is never sent to the model.

const (
	JobModeFull       = "FULL"
	JobModeSupplement = "SUPPLEMENT"

	// A batch is kept well under the single-call limits (80 excerpts / 40k chars)
	// so one Gemini call finishes in about a minute.
	BatchMaxExcerpts = 40
	BatchMaxChars    = 16000
	// MaxJobBatches bounds one job (~1200 excerpts). The largest section of the
	// 451-page decision needs 10 batches; anything bigger is not one procedure.
	MaxJobBatches = 30
)

// PlanBlock reasons.
const (
	PlanSectionNotFound  = "SECTION_NOT_FOUND"
	PlanSectionAmbiguous = "SECTION_AMBIGUOUS"
	PlanMixedProcedures  = "MIXED_PROCEDURES"
	PlanScopeTooLarge    = "SCOPE_TOO_LARGE"
	PlanNothingNew       = "NOTHING_NEW"
	PlanBaseMismatch     = "BASE_SOURCE_MISMATCH"
	PlanEmptyScope       = "EMPTY_SCOPE"
	PlanOutsideSection   = "OUTSIDE_SECTION"
)

var (
	ErrPlanBlocked     = errors.New("drafting: plan blocked")
	ErrJobNotRetryable = errors.New("drafting: job is not in a retryable state")
)

// JobStore is the job persistence surface (*repository.DraftJobRepo).
type JobStore interface {
	Create(ctx context.Context, job repository.DraftJob, batches []repository.DraftJobBatch) (repository.DraftJob, bool, error)
	Get(ctx context.Context, xaID string, jobID uuid.UUID) (repository.DraftJob, []repository.DraftJobBatch, error)
	List(ctx context.Context, xaID string, procedureID uuid.UUID, limit int) ([]repository.DraftJob, error)
	Claim(ctx context.Context, xaID string, owner uuid.UUID, lease time.Duration) (repository.DraftJob, bool, error)
	ExtendLease(ctx context.Context, jobID, owner uuid.UUID, lease time.Duration) (bool, error)
	StartBatch(ctx context.Context, jobID, owner uuid.UUID, index int) error
	EndBatch(ctx context.Context, jobID, owner uuid.UUID, index int, status string, result json.RawMessage, code, message string) error
	Finish(ctx context.Context, jobID, owner uuid.UUID, status string, draftID *uuid.UUID, code, message string) error
	Retry(ctx context.Context, xaID string, jobID uuid.UUID) (repository.DraftJob, error)
}

type DraftPlanRequest struct {
	DocumentID   uuid.UUID  `json:"document_id"`
	GenerationID uuid.UUID  `json:"generation_id"`
	SectionIndex *int       `json:"section_index"`
	PageStart    int        `json:"page_start"`
	PageEnd      int        `json:"page_end"`
	Mode         string     `json:"mode"`
	BaseDraftID  *uuid.UUID `json:"base_draft_id"`
}

type PlanBatch struct {
	Index      int         `json:"index"`
	PageStart  int         `json:"page_start"`
	PageEnd    int         `json:"page_end"`
	ChunkStart int         `json:"chunk_start"`
	ChunkEnd   int         `json:"chunk_end"`
	Excerpts   int         `json:"excerpts"`
	Chars      int         `json:"chars"`
	ExcerptIDs []uuid.UUID `json:"-"`
}

type DraftPlan struct {
	ProcedureID       uuid.UUID    `json:"procedure_id"`
	ProcedureName     string       `json:"procedure_name"`
	DocumentID        uuid.UUID    `json:"document_id"`
	GenerationID      uuid.UUID    `json:"generation_id"`
	DocumentPages     int          `json:"document_pages"`
	DocumentChunks    int          `json:"document_chunks"`
	Sections          []Section    `json:"sections"`
	SelectedSection   *int         `json:"selected_section,omitempty"`
	Selection         string       `json:"selection"`
	SectionTitle      string       `json:"section_title,omitempty"`
	BoundaryUncertain bool         `json:"boundary_uncertain,omitempty"`
	PageStart         int          `json:"page_start"`
	PageEnd           int          `json:"page_end"`
	ChunkRanges       []ChunkRange `json:"chunk_ranges"`
	SelectedChunks    int          `json:"selected_chunks"`
	SelectedChars     int          `json:"selected_chars"`
	Mode              string       `json:"mode"`
	BaseDraftID       *uuid.UUID   `json:"base_draft_id,omitempty"`
	BaseChunkRanges   []ChunkRange `json:"base_chunk_ranges,omitempty"`
	BaseChunks        int          `json:"base_chunks,omitempty"`
	NewChunks         int          `json:"new_chunks"`
	Batches           []PlanBatch  `json:"batches"`
	Ready             bool         `json:"ready"`
	BlockedReason     string       `json:"blocked_reason,omitempty"`
	Warnings          []string     `json:"warnings"`
	// OutsideMentions: chunks outside the planned scope (selection + base) that
	// name the procedure, grouped by section. Listed for the Admin, never merged.
	OutsideMentions []OutsideMention `json:"outside_mentions"`
}

// DraftJobView is a job with its batch progress (batch results are not exposed).
type DraftJobView struct {
	repository.DraftJob
	Batches []repository.DraftJobBatch `json:"batches"`
}

// PlanDraft finds the procedure's section in the whole indexed document and
// splits it into batches. It persists nothing.
func (s *Service) PlanDraft(ctx context.Context, procedureID uuid.UUID, req DraftPlanRequest) (DraftPlan, error) {
	if s == nil || s.Source == nil || s.Drafts == nil {
		return DraftPlan{}, ErrNotConfigured
	}
	if req.DocumentID == uuid.Nil {
		return DraftPlan{}, fmt.Errorf("drafting: document_id is required")
	}
	mode := strings.ToUpper(strings.TrimSpace(req.Mode))
	if mode == "" {
		mode = JobModeFull
	}
	if mode != JobModeFull && mode != JobModeSupplement {
		return DraftPlan{}, fmt.Errorf("drafting: invalid mode")
	}
	if (mode == JobModeSupplement) != (req.BaseDraftID != nil) {
		return DraftPlan{}, fmt.Errorf("drafting: base_draft_id is required for SUPPLEMENT and only for SUPPLEMENT")
	}
	if (req.PageStart > 0) != (req.PageEnd > 0) || req.PageEnd < req.PageStart {
		return DraftPlan{}, fmt.Errorf("drafting: invalid page range")
	}
	generationID := req.GenerationID
	if generationID == uuid.Nil {
		sources, err := s.Source.ListReady(ctx, s.XAID, procedureID)
		if err != nil {
			return DraftPlan{}, err
		}
		for _, src := range sources {
			if src.DocumentID == req.DocumentID {
				generationID = src.GenerationID
				break
			}
		}
		if generationID == uuid.Nil {
			return DraftPlan{}, ErrSourceNotFound
		}
	}
	all, err := s.Source.ResolveChunks(ctx, s.XAID, procedureID, req.DocumentID, generationID, nil)
	if err != nil {
		return DraftPlan{}, err
	}
	plan := DraftPlan{
		ProcedureID: procedureID, ProcedureName: all.ProcedureName,
		DocumentID: req.DocumentID, GenerationID: generationID,
		DocumentChunks: len(all.Excerpts), Mode: mode, BaseDraftID: req.BaseDraftID,
		Warnings: []string{}, ChunkRanges: []ChunkRange{}, Batches: []PlanBatch{},
	}
	_, plan.DocumentPages = excerptPageBounds(all.Excerpts)
	plan.Sections = DetectSections(all.Excerpts)
	best := ScoreSections(plan.Sections, all.ProcedureName)

	block := func(reason, warning string) (DraftPlan, error) {
		plan.BlockedReason = reason
		if warning != "" {
			plan.Warnings = append(plan.Warnings, warning)
		}
		return plan, nil
	}

	var selected []Excerpt
	switch {
	case req.PageStart > 0:
		plan.Selection = "manual"
		for _, ex := range all.Excerpts {
			if ex.PageStart >= req.PageStart && ex.PageEnd <= req.PageEnd {
				selected = append(selected, ex)
			}
		}
		if foreign := foreignSections(plan.Sections, selected); len(foreign) > 0 {
			return block(PlanMixedProcedures, "Phạm vi trang chứa nội dung của thủ tục khác: "+strings.Join(foreign, "; ")+". Chọn đúng mục của thủ tục này.")
		}
		if outsideOwnSection(plan.Sections, selected) {
			if mode == JobModeFull {
				return block(PlanOutsideSection, "Khoảng trang không chứa đoạn nào của mục \""+exactTitle(plan.Sections)+"\" — không soạn định nghĩa từ trang ngoài mục thủ tục. Dùng \"Tự tìm mục\".")
			}
			plan.Warnings = append(plan.Warnings, "Khoảng trang nằm ngoài mục \""+exactTitle(plan.Sections)+"\" (ví dụ bảng danh mục). Chỉ bổ sung khi chắc chắn trang này quy định riêng cho thủ tục.")
		} else if w := partialSectionWarning(plan.Sections, selected); w != "" {
			plan.Warnings = append(plan.Warnings, w)
		}
	case req.SectionIndex != nil:
		idx := *req.SectionIndex
		if idx < 0 || idx >= len(plan.Sections) {
			return DraftPlan{}, fmt.Errorf("drafting: invalid section_index")
		}
		plan.Selection = "section"
		plan.SelectedSection = &idx
		if !plan.Sections[idx].ExactMatch {
			plan.Warnings = append(plan.Warnings, fmt.Sprintf("Tên mục \"%s\" khác tên thủ tục \"%s\" — Admin đã tự chọn.", plan.Sections[idx].Title, all.ProcedureName))
		}

	case best == -2:
		plan.Selection = "none"
		return block(PlanSectionAmbiguous, "Có nhiều mục cùng tên thủ tục. Chọn một mục cụ thể.")
	case best < 0:
		plan.Selection = "none"
		return block(PlanSectionNotFound, "Không tìm thấy mục nào có đúng tên thủ tục \""+all.ProcedureName+"\". Chọn mục hoặc nhập khoảng trang.")
	default:
		plan.Selection = "section"
		plan.SelectedSection = &best
	}
	if plan.SelectedSection != nil {
		sec := plan.Sections[*plan.SelectedSection]
		plan.SectionTitle = sec.Title
		plan.BoundaryUncertain = sec.BoundaryUncertain
		if sec.BoundaryUncertain {
			plan.Warnings = append(plan.Warnings, "Không chắc ranh giới cuối mục (số thứ tự mục kế tiếp bị nhảy) — kiểm tra trang cuối.")
		}
		for _, ex := range all.Excerpts {
			if ex.ChunkIndex >= sec.ChunkStart && ex.ChunkIndex <= sec.ChunkEnd {
				selected = append(selected, ex)
			}
		}
	}
	if len(selected) == 0 {
		return block(PlanEmptyScope, "Phạm vi đã chọn không có đoạn văn bản nào.")
	}
	indexes := make([]int, 0, len(selected))
	for _, ex := range selected {
		indexes = append(indexes, ex.ChunkIndex)
		plan.SelectedChars += len([]rune(ex.Content))
	}
	plan.ChunkRanges = chunkRangesOf(indexes)
	plan.SelectedChunks = len(selected)
	plan.PageStart, plan.PageEnd = excerptPageBounds(selected)

	work := selected
	if mode == JobModeSupplement {
		base, err := s.Get(ctx, *req.BaseDraftID)
		if err != nil {
			return DraftPlan{}, err
		}
		if base.ProcedureID != procedureID || base.DocumentID != req.DocumentID || base.GenerationID != generationID {
			return block(PlanBaseMismatch, "Bản nháp nền dùng tài liệu hoặc generation khác — không gộp được.")
		}
		plan.BaseChunkRanges = baseChunkRanges(base, all.Excerpts)
		work = work[:0:0]
		for _, ex := range selected {
			if inRanges(ex.ChunkIndex, plan.BaseChunkRanges) {
				continue
			}
			work = append(work, ex)
		}
		for _, ex := range all.Excerpts {
			if inRanges(ex.ChunkIndex, plan.BaseChunkRanges) {
				plan.BaseChunks++
			}
		}
		if len(work) == 0 {
			return block(PlanNothingNew, "Bản nháp nền đã bao phủ toàn bộ phạm vi này.")
		}
		if w := uncoveredSectionWarning(plan.Sections, all.Excerpts, plan.BaseChunkRanges, selected); w != "" {
			plan.Warnings = append(plan.Warnings, w)
		}
	}
	plan.OutsideMentions = outsideMentions(plan.Sections, all.Excerpts, all.ProcedureName, append(append([]ChunkRange{}, plan.ChunkRanges...), plan.BaseChunkRanges...))
	for _, m := range plan.OutsideMentions {
		switch {
		case m.Section == "" && m.Kind == "name":
			plan.Warnings = append(plan.Warnings, fmt.Sprintf("Ngoài các mục thủ tục có %d đoạn nhắc \"%s\" (trang %d–%d), chưa đưa vào bản nháp — Admin kiểm tra.", m.Chunks, all.ProcedureName, m.PageStart, m.PageEnd))
		case m.Section == "":
			plan.Warnings = append(plan.Warnings, fmt.Sprintf("Ngoài các mục thủ tục có %d đoạn có từ khóa quy định chung (lệ phí, thời hạn, biểu mẫu…) ở trang %d–%d — Admin kiểm tra có áp dụng cho thủ tục này không.", m.Chunks, m.PageStart, m.PageEnd))
		}
	}
	plan.NewChunks = len(work)
	plan.Batches = splitBatches(work)
	if len(plan.Batches) > MaxJobBatches {
		return block(PlanScopeTooLarge, fmt.Sprintf("Phạm vi cần %d đợt (tối đa %d). Thu hẹp về đúng mục của thủ tục.", len(plan.Batches), MaxJobBatches))
	}
	plan.Ready = true
	return plan, nil
}

// foreignSections names other procedures' sections that a manual range covers
// beyond the single chunk two sections can share at their boundary.
func foreignSections(sections []Section, selected []Excerpt) []string {
	out := make([]string, 0)
	for _, sec := range sections {
		if sec.ExactMatch {
			continue
		}
		n := 0
		for _, ex := range selected {
			if ex.ChunkIndex >= sec.ChunkStart && ex.ChunkIndex <= sec.ChunkEnd {
				n++
			}
		}
		if n >= 2 {
			out = append(out, sec.Title)
		}
	}
	return out
}

func exactTitle(sections []Section) string {
	for _, sec := range sections {
		if sec.ExactMatch {
			return sec.Title
		}
	}
	return ""
}

// outsideOwnSection: the procedure has its own section but the selection has
// none of its chunks (e.g. only the Part I summary table).
func outsideOwnSection(sections []Section, selected []Excerpt) bool {
	has := false
	for _, sec := range sections {
		if !sec.ExactMatch {
			continue
		}
		has = true
		for _, ex := range selected {
			if ex.ChunkIndex >= sec.ChunkStart && ex.ChunkIndex <= sec.ChunkEnd {
				return false
			}
		}
	}
	return has
}

// uncoveredSectionWarning: base draft + new range still leave part of the
// procedure's own section undrafted.
func uncoveredSectionWarning(sections []Section, all []Excerpt, base []ChunkRange, selected []Excerpt) string {
	chosen := map[int]bool{}
	for _, ex := range selected {
		chosen[ex.ChunkIndex] = true
	}
	for _, sec := range sections {
		if !sec.ExactMatch {
			continue
		}
		missing := make([]int, 0)
		pages := map[int]bool{}
		for _, ex := range all {
			if ex.ChunkIndex < sec.ChunkStart || ex.ChunkIndex > sec.ChunkEnd || chosen[ex.ChunkIndex] || inRanges(ex.ChunkIndex, base) {
				continue
			}
			missing = append(missing, ex.ChunkIndex)
			pages[ex.PageStart] = true
		}
		if len(missing) > 0 {
			list := make([]int, 0, len(pages))
			for p := range pages {
				list = append(list, p)
			}
			sort.Ints(list)
			return fmt.Sprintf("Bản nền + phạm vi mới vẫn chưa phủ %d/%d đoạn của mục \"%s\" (trang %s). Nên dùng \"Soạn mới toàn bộ\" + \"Tự tìm mục\".", len(missing), sec.Chunks, sec.Title, joinInts(list))
		}
	}
	return ""
}

// OutsideCriteria describes the outside search, stored with each coverage report.
const OutsideCriteria = "Các đoạn ngoài phạm vi có tên thủ tục; và các đoạn ngoài mọi mục thủ tục có từ khóa quy định chung (lệ phí, phí, thời hạn, biểu mẫu/mẫu, quy định chung, áp dụng chung). Không chứng minh tài liệu không còn quy định khác."

// generalRuleRe: general-provision keywords for chunks outside every section.
var generalRuleRe = regexp.MustCompile(`(?i)(?:^|[^\p{L}])(?:lệ\s*phí|phí|thời\s*hạn|biểu\s*mẫu|mẫu\s+(?:số|tờ)|quy\s*định\s*chung|áp\s*dụng\s*chung)(?:$|[^\p{L}])`)

// outsideMentions lists chunks outside scope that contain the procedure name
// (any section) and, outside every procedure section, chunks with general-rule
// keywords that may apply to all procedures. Results are only reported.
func outsideMentions(sections []Section, all []Excerpt, procedureName string, scope []ChunkRange) []OutsideMention {
	want := normText(procedureName)
	out := make([]OutsideMention, 0)
	index := map[string]int{}
	for _, ex := range all {
		if inRanges(ex.ChunkIndex, scope) {
			continue
		}
		title := ""
		for _, sec := range sections {
			if ex.ChunkIndex >= sec.ChunkStart && ex.ChunkIndex <= sec.ChunkEnd {
				title = sec.Title
				break
			}
		}
		kind := ""
		switch {
		case want != "" && strings.Contains(normText(ex.Content), want):
			kind = "name"
		case title == "" && generalRuleRe.MatchString(ex.Content):
			kind = "general"
		default:
			continue
		}
		key := kind + "|" + title
		i, ok := index[key]
		if !ok {
			i = len(out)
			index[key] = i
			out = append(out, OutsideMention{Kind: kind, Section: title, PageStart: ex.PageStart})
		}
		out[i].Chunks++
		out[i].ChunkIndexes = append(out[i].ChunkIndexes, ex.ChunkIndex)
		out[i].Pages = addPageSpan(out[i].Pages, ex.PageStart, ex.PageEnd)
		if ex.PageEnd > out[i].PageEnd {
			out[i].PageEnd = ex.PageEnd
		}
	}
	return out
}

// addPageSpan merges a page range into ordered spans (chunks arrive in order).
func addPageSpan(spans []PageSpan, start, end int) []PageSpan {
	if n := len(spans); n > 0 && start <= spans[n-1].End+1 {
		if end > spans[n-1].End {
			spans[n-1].End = end
		}
		return spans
	}
	return append(spans, PageSpan{Start: start, End: end})
}

func joinInts(xs []int) string {
	parts := make([]string, 0, len(xs))
	for _, x := range xs {
		parts = append(parts, fmt.Sprint(x))
	}
	return strings.Join(parts, ", ")
}

// partialSectionWarning: a manual range that covers only part of the
// procedure's own section yields an incomplete Definition.
func partialSectionWarning(sections []Section, selected []Excerpt) string {
	for _, sec := range sections {
		if !sec.ExactMatch {
			continue
		}
		n := 0
		for _, ex := range selected {
			if ex.ChunkIndex >= sec.ChunkStart && ex.ChunkIndex <= sec.ChunkEnd {
				n++
			}
		}
		if n > 0 && n < sec.Chunks {
			return fmt.Sprintf("Khoảng trang chỉ gồm %d/%d đoạn của mục \"%s\" (trang %d–%d) — định nghĩa có thể thiếu. Nên dùng \"Tự tìm mục\".", n, sec.Chunks, sec.Title, sec.PageStart, sec.PageEnd)
		}
	}
	return ""
}

// baseChunkRanges: chunk ranges of the base draft, or its page range for a
// single-call draft that predates chunk ranges.
func baseChunkRanges(base DraftView, all []Excerpt) []ChunkRange {
	if len(base.ChunkRanges) > 0 {
		return base.ChunkRanges
	}
	idx := make([]int, 0)
	for _, ex := range all {
		if ex.PageStart >= base.SourcePageStart && ex.PageEnd <= base.SourcePageEnd {
			idx = append(idx, ex.ChunkIndex)
		}
	}
	return chunkRangesOf(idx)
}

// splitBatches cuts ordered excerpts into contiguous batches within the limits.
func splitBatches(items []Excerpt) []PlanBatch {
	out := make([]PlanBatch, 0)
	var cur *PlanBatch
	lastChunk := -2
	for _, ex := range items {
		chars := len([]rune(ex.Content))
		if cur == nil || cur.Excerpts >= BatchMaxExcerpts || cur.Chars+chars > BatchMaxChars || ex.ChunkIndex != lastChunk+1 {
			out = append(out, PlanBatch{Index: len(out), PageStart: ex.PageStart, ChunkStart: ex.ChunkIndex})
			cur = &out[len(out)-1]
		}
		cur.Excerpts++
		cur.Chars += chars
		cur.ChunkEnd = ex.ChunkIndex
		if ex.PageEnd > cur.PageEnd {
			cur.PageEnd = ex.PageEnd
		}
		if ex.PageStart < cur.PageStart {
			cur.PageStart = ex.PageStart
		}
		cur.ExcerptIDs = append(cur.ExcerptIDs, ex.ExcerptID)
		lastChunk = ex.ChunkIndex
	}
	return out
}

// CreateDraftJob plans and queues a job. Same X-Request-ID + same body replays.
func (s *Service) CreateDraftJob(ctx context.Context, actor, requestID, procedureID uuid.UUID, req DraftPlanRequest) (DraftJobView, bool, error) {
	if s.Jobs == nil {
		return DraftJobView{}, false, ErrNotConfigured
	}
	plan, err := s.PlanDraft(ctx, procedureID, req)
	if err != nil {
		return DraftJobView{}, false, err
	}
	if !plan.Ready {
		return DraftJobView{}, false, fmt.Errorf("%w: %s", ErrPlanBlocked, plan.BlockedReason)
	}
	body, _ := json.Marshal(req)
	sum := sha256.Sum256(body)
	stored := plan
	stored.Sections = nil
	planRaw, err := json.Marshal(stored)
	if err != nil {
		return DraftJobView{}, false, err
	}
	batches := make([]repository.DraftJobBatch, 0, len(plan.Batches))
	for _, b := range plan.Batches {
		batches = append(batches, repository.DraftJobBatch{
			BatchIndex: b.Index, PageStart: b.PageStart, PageEnd: b.PageEnd,
			ChunkStart: b.ChunkStart, ChunkEnd: b.ChunkEnd, ExcerptIDs: b.ExcerptIDs,
		})
	}
	job, created, err := s.Jobs.Create(ctx, repository.DraftJob{
		XaID: s.XAID, ProcedureID: procedureID, DocumentID: plan.DocumentID, GenerationID: plan.GenerationID,
		Mode: plan.Mode, BaseDraftID: plan.BaseDraftID, RequestID: requestID,
		RequestHash: hex.EncodeToString(sum[:]), Plan: planRaw, CreatedBy: actor,
	}, batches)
	if err != nil {
		return DraftJobView{}, false, err
	}
	if s.Runner != nil {
		s.Runner.Wake()
	}
	view, err := s.GetDraftJob(ctx, job.ID)
	return view, created, err
}

func (s *Service) GetDraftJob(ctx context.Context, jobID uuid.UUID) (DraftJobView, error) {
	if s.Jobs == nil {
		return DraftJobView{}, ErrNotConfigured
	}
	job, batches, err := s.Jobs.Get(ctx, s.XAID, jobID)
	if err != nil {
		return DraftJobView{}, err
	}
	return DraftJobView{DraftJob: job, Batches: batches}, nil
}

func (s *Service) ListDraftJobs(ctx context.Context, procedureID uuid.UUID) ([]repository.DraftJob, error) {
	if s.Jobs == nil {
		return nil, ErrNotConfigured
	}
	return s.Jobs.List(ctx, s.XAID, procedureID, 20)
}

// RetryDraftJob re-queues the failed batches (or the merge) of a FAILED job.
func (s *Service) RetryDraftJob(ctx context.Context, jobID uuid.UUID) (DraftJobView, error) {
	if s.Jobs == nil {
		return DraftJobView{}, ErrNotConfigured
	}
	if _, err := s.Jobs.Retry(ctx, s.XAID, jobID); err != nil {
		if errors.Is(err, repository.ErrConflict) {
			return DraftJobView{}, ErrJobNotRetryable
		}
		return DraftJobView{}, err
	}
	if s.Runner != nil {
		s.Runner.Wake()
	}
	return s.GetDraftJob(ctx, jobID)
}

// JobRunner runs queued draft jobs in the background, one at a time per process.
// A lease with heartbeat lets another process (or this one after a restart)
// resume a job whose worker died; finished batches are never re-run.
type JobRunner struct {
	Service     *Service
	Owner       uuid.UUID
	Lease       time.Duration
	Poll        time.Duration
	MaxAttempts int
	Backoff     func(attempt int) time.Duration
	wake        chan struct{}
	start       sync.Once
}

func NewJobRunner(s *Service) *JobRunner {
	return &JobRunner{
		Service: s, Owner: uuid.New(), Lease: 3 * time.Minute, Poll: 15 * time.Second, MaxAttempts: 3,
		Backoff: func(attempt int) time.Duration { return time.Duration(attempt) * 10 * time.Second },
		wake:    make(chan struct{}, 1),
	}
}

// Wake starts the background loop on first use and asks it to look for work.
func (r *JobRunner) Wake() {
	r.start.Do(func() { r.Start(context.Background()) })
	select {
	case r.wake <- struct{}{}:
	default:
	}
}

func (r *JobRunner) Start(ctx context.Context) {
	go func() {
		ticker := time.NewTicker(r.Poll)
		defer ticker.Stop()
		for {
			r.Drain(ctx)
			select {
			case <-ctx.Done():
				return
			case <-r.wake:
			case <-ticker.C:
			}
		}
	}()
}

// Drain runs claimable jobs until none is left.
func (r *JobRunner) Drain(ctx context.Context) {
	for ctx.Err() == nil {
		job, ok, err := r.Service.Jobs.Claim(ctx, r.Service.XAID, r.Owner, r.Lease)
		if err != nil {
			log.Printf("draft_job_claim error=%v", err)
			return
		}
		if !ok {
			return
		}
		r.runLeased(ctx, job)
	}
}

func (r *JobRunner) runLeased(parent context.Context, job repository.DraftJob) {
	ctx, cancel := context.WithCancel(parent)
	defer cancel()
	go func() {
		t := time.NewTicker(r.Lease / 3)
		defer t.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-t.C:
				ok, err := r.Service.Jobs.ExtendLease(ctx, job.ID, r.Owner, r.Lease)
				if err == nil && !ok {
					log.Printf("draft_job_lease_lost job=%s", job.ID)
					cancel()
					return
				}
			}
		}
	}()
	if err := r.run(ctx, job); err != nil && ctx.Err() == nil {
		log.Printf("draft_job_error job=%s error=%v", job.ID, err)
	}
}

func (r *JobRunner) run(ctx context.Context, job repository.DraftJob) error {
	s := r.Service
	var plan DraftPlan
	if err := json.Unmarshal(job.Plan, &plan); err != nil {
		return s.Jobs.Finish(ctx, job.ID, r.Owner, repository.DraftJobFailed, nil, "PLAN_INVALID", err.Error())
	}
	_, batches, err := s.Jobs.Get(ctx, s.XAID, job.ID)
	if err != nil {
		return err
	}
	known := []KnownSlot{}
	for _, b := range batches {
		if b.Status == repository.DraftBatchSucceeded {
			known = mergeKnownSlots(known, b.Result)
			continue
		}
		if b.Status == repository.DraftBatchFailed {
			continue
		}
		resp, code, msg := r.runBatch(ctx, job, plan, b, len(batches), known)
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if code != "" {
			log.Printf("draft_job_batch_failed job=%s batch=%d code=%s message=%s", job.ID, b.BatchIndex, code, msg)
			continue
		}
		known = mergeKnownSlots(known, resp)
	}
	_, batches, err = s.Jobs.Get(ctx, s.XAID, job.ID)
	if err != nil {
		return err
	}
	failed := make([]string, 0)
	for _, b := range batches {
		if b.Status != repository.DraftBatchSucceeded {
			failed = append(failed, fmt.Sprintf("đợt %d (trang %d–%d)", b.BatchIndex+1, b.PageStart, b.PageEnd))
		}
	}
	if len(failed) > 0 {
		// Partial results are kept; no draft is created from part of the source.
		return s.Jobs.Finish(ctx, job.ID, r.Owner, repository.DraftJobFailed, nil, "BATCH_FAILED",
			fmt.Sprintf("%d/%d đợt chưa xong: %s", len(failed), len(batches), strings.Join(failed, ", ")))
	}
	draftID, err := s.aggregateJob(ctx, job, plan, batches)
	if err != nil {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		return s.Jobs.Finish(ctx, job.ID, r.Owner, repository.DraftJobFailed, nil, "AGGREGATION_FAILED", err.Error())
	}
	return s.Jobs.Finish(ctx, job.ID, r.Owner, repository.DraftJobSucceeded, &draftID, "", "")
}

// runBatch drafts one batch with retries; returns the stored result or a final error code.
func (r *JobRunner) runBatch(ctx context.Context, job repository.DraftJob, plan DraftPlan, b repository.DraftJobBatch, total int, known []KnownSlot) (json.RawMessage, string, string) {
	s := r.Service
	attempts := b.Attempts
	for {
		if attempts >= r.MaxAttempts {
			code, msg := "ATTEMPTS_EXHAUSTED", "no attempts left"
			if b.ErrorCode != nil {
				code = *b.ErrorCode
			}
			if b.ErrorMessage != nil {
				msg = *b.ErrorMessage
			}
			if err := s.Jobs.StartBatch(ctx, job.ID, r.Owner, b.BatchIndex); err == nil {
				_ = s.Jobs.EndBatch(ctx, job.ID, r.Owner, b.BatchIndex, repository.DraftBatchFailed, nil, code, msg)
			}
			return nil, code, msg
		}
		if err := s.Jobs.StartBatch(ctx, job.ID, r.Owner, b.BatchIndex); err != nil {
			return nil, "LEASE_LOST", err.Error()
		}
		attempts++
		raw, code, msg, retryable := s.draftOneBatch(ctx, job, plan, b, total, known, attempts)
		if ctx.Err() != nil {
			return nil, "CANCELLED", "worker stopped"
		}
		if code == "" {
			if err := s.Jobs.EndBatch(ctx, job.ID, r.Owner, b.BatchIndex, repository.DraftBatchSucceeded, raw, "", ""); err != nil {
				return nil, "LEASE_LOST", err.Error()
			}
			return raw, "", ""
		}
		final := !retryable || attempts >= r.MaxAttempts
		status := repository.DraftBatchPending
		if final {
			status = repository.DraftBatchFailed
		}
		if err := s.Jobs.EndBatch(ctx, job.ID, r.Owner, b.BatchIndex, status, nil, code, msg); err != nil {
			return nil, "LEASE_LOST", err.Error()
		}
		if final {
			return nil, code, msg
		}
		select {
		case <-ctx.Done():
			return nil, "CANCELLED", "worker stopped"
		case <-time.After(r.Backoff(attempts)):
		}
	}
}

func (s *Service) draftOneBatch(ctx context.Context, job repository.DraftJob, plan DraftPlan, b repository.DraftJobBatch, total int, known []KnownSlot, attempt int) (json.RawMessage, string, string, bool) {
	snap, err := s.Source.ResolveChunks(ctx, s.XAID, job.ProcedureID, job.DocumentID, job.GenerationID, []ChunkRange{{Start: b.ChunkStart, End: b.ChunkEnd}})
	if err != nil {
		return nil, "SOURCE_UNAVAILABLE", err.Error(), false
	}
	want := map[uuid.UUID]bool{}
	for _, id := range b.ExcerptIDs {
		want[id] = true
	}
	kept := snap.Excerpts[:0:0]
	for _, ex := range snap.Excerpts {
		if want[ex.ExcerptID] {
			kept = append(kept, ex)
		}
	}
	if len(kept) != len(b.ExcerptIDs) {
		return nil, "SOURCE_CHANGED", "planned excerpts are no longer in the verified generation", false
	}
	snap.Excerpts = kept
	snap.SourcePageStart, snap.SourcePageEnd = excerptPageBounds(kept)
	reqID := uuid.NewSHA1(job.ID, []byte(fmt.Sprintf("batch-%d-attempt-%d", b.BatchIndex, attempt)))
	resp, err := s.Client.DraftBatch(ctx, reqID, snap, &BatchContext{
		Partial: true, SectionTitle: plan.SectionTitle,
		BatchIndex: b.BatchIndex + 1, BatchCount: total, KnownSlots: known,
	})
	if err != nil {
		switch {
		case errors.Is(err, ErrDraftTimeout):
			return nil, "DRAFT_TIMEOUT", err.Error(), true
		case errors.Is(err, ErrRateLimited):
			return nil, "RATE_LIMITED", err.Error(), true
		case errors.Is(err, ErrInvalidDraft):
			return nil, "AI_INVALID_OUTPUT", err.Error(), true
		default:
			return nil, "AI_PROVIDER", err.Error(), true
		}
	}
	raw, err := json.Marshal(resp)
	if err != nil {
		return nil, "AI_INVALID_OUTPUT", err.Error(), true
	}
	return raw, "", "", false
}

func mergeKnownSlots(known []KnownSlot, raw json.RawMessage) []KnownSlot {
	var resp ProviderResponse
	if json.Unmarshal(raw, &resp) != nil {
		return known
	}
	var def aggDef
	if json.Unmarshal(resp.Definition, &def) != nil {
		return known
	}
	have := map[string]bool{}
	for _, k := range known {
		have[k.Key] = true
	}
	for _, key := range sortedSlotKeys(def.Slots) {
		if have[key] || len(known) >= aggMaxSlots {
			continue
		}
		slot := def.Slots[key]
		known = append(known, KnownSlot{Key: key, Type: slot.Type, Question: slot.Question})
		have[key] = true
	}
	return known
}

// aggregateJob merges the stored batch results into a NEW draft row. The draft
// id is derived from the job id, so re-running the merge never duplicates it.
func (s *Service) aggregateJob(ctx context.Context, job repository.DraftJob, plan DraftPlan, batches []repository.DraftJobBatch) (uuid.UUID, error) {
	draftID := uuid.NewSHA1(job.ID, []byte("aggregate-draft"))
	if existing, err := s.Drafts.Get(ctx, s.XAID, draftID); err == nil {
		return existing.ID, nil
	}
	ranges := append([]ChunkRange{}, plan.ChunkRanges...)
	contributions := make([]Contribution, 0, len(batches)+1)
	var base *DraftView
	if job.Mode == JobModeSupplement && job.BaseDraftID != nil {
		b, err := s.Get(ctx, *job.BaseDraftID)
		if err != nil {
			return uuid.Nil, err
		}
		base = &b
		ranges = append(ranges, plan.BaseChunkRanges...)
		contributions = append(contributions, Contribution{
			Label: "base", PageStart: b.SourcePageStart, PageEnd: b.SourcePageEnd,
			Definition: b.Definition, Evidence: b.Evidence, Warnings: b.Warnings, Missing: b.MissingRequirements,
			OpenConflicts: b.Conflicts, Prefixed: true,
		})
	}
	snap, err := s.Source.ResolveChunks(ctx, s.XAID, job.ProcedureID, job.DocumentID, job.GenerationID, ranges)
	if err != nil {
		return uuid.Nil, err
	}
	var first *ProviderResponse
	usage := map[string]any{}
	latency, processed := 0, 0
	mismatch := false
	for _, b := range batches {
		var resp ProviderResponse
		if err := json.Unmarshal(b.Result, &resp); err != nil {
			return uuid.Nil, fmt.Errorf("batch %d result: %w", b.BatchIndex, err)
		}
		if first == nil {
			r := resp
			first = &r
		}
		mismatch = mismatch || resp.IdentityMismatch
		latency += resp.LatencyMS
		addUsage(usage, resp.Usage)
		processed += len(b.ExcerptIDs)
		contributions = append(contributions, Contribution{
			Label: fmt.Sprintf("batch:%d", b.BatchIndex+1), PageStart: b.PageStart, PageEnd: b.PageEnd,
			Definition: resp.Definition, Evidence: resp.Evidence, Warnings: resp.Warnings, Missing: resp.MissingRequirements,
		})
	}
	agg, err := Aggregate(snap.ProcedureCode, snap.ProcedureName, contributions, snap.Excerpts)
	if err != nil {
		return uuid.Nil, err
	}
	locked, _, _, lockErr := lockProcedureIdentity(agg.Definition, snap.ProcedureCode, snap.ProcedureName)
	if lockErr != nil && locked == nil {
		return uuid.Nil, lockErr
	}
	goVal := decision.ValidateDraftDefinition(locked, snap.ProcedureCode)
	cov := EvaluateCoverage(locked, agg.Evidence, snap.Excerpts, snap)
	applyCoverage(&goVal, cov)
	if mismatch {
		applyIdentityFinding(&goVal, "", snap.ProcedureCode, true)
	}
	jobID := job.ID
	coverage := &DraftCoverage{
		Mode: job.Mode, JobID: job.ID, BaseDraftID: job.BaseDraftID,
		Selection: plan.Selection, SectionTitle: plan.SectionTitle, BoundaryUncertain: plan.BoundaryUncertain,
		DocumentPages: plan.DocumentPages, DocumentChunks: plan.DocumentChunks,
		PageStart: snap.SourcePageStart, PageEnd: snap.SourcePageEnd,
		SelectedChunks: plan.NewChunks, BaseChunks: plan.BaseChunks, ProcessedChunks: processed,
		Batches: len(batches), ScopeComplete: processed == plan.NewChunks, Complete: processed == plan.NewChunks,
		Truncated:         agg.Truncated,
		DuplicatesRemoved: agg.DuplicatesRemoved,
		OutsideMentions:   plan.OutsideMentions,
		OutsideCriteria:   OutsideCriteria,
	}
	view := DraftView{
		ID:               draftID,
		ProcedureID:      job.ProcedureID,
		ProcedureCode:    snap.ProcedureCode,
		DocumentID:       job.DocumentID,
		Status:           repository.DraftStatusDraft,
		Definition:       locked,
		Evidence:         cov.Evidence,
		Proposals:        cov.Proposals,
		Warnings:         agg.Warnings,
		Validation:       goVal,
		GenerationID:     job.GenerationID,
		SourcePageStart:  snap.SourcePageStart,
		SourcePageEnd:    snap.SourcePageEnd,
		Usage:            usage,
		LatencyMS:        latency,
		CreatedBy:        &job.CreatedBy,
		IdentityMismatch: mismatch,
		Revision:         1,
		JobID:            &jobID,
		ChunkRanges:      chunkRangesOf(chunkIndexes(snap.Excerpts)),
		Coverage:         coverage,
		Conflicts:        agg.Conflicts,
		Provenance:       agg.Provenance,
	}
	view.MissingRequirements = agg.Missing
	if base != nil {
		view.ParentDraftID = &base.ID
	}
	if first != nil {
		view.Model, view.Provider, view.PromptVersion, view.PromptHash = first.Model, first.Provider, first.PromptVersion, first.PromptHash
	}
	applyAggregateFindings(&view, snap.Excerpts)
	payload, err := json.Marshal(storedPayloadFromView(view))
	if err != nil {
		return uuid.Nil, err
	}
	validationRaw, err := json.Marshal(view.Validation)
	if err != nil {
		return uuid.Nil, err
	}
	saved, err := s.Drafts.Insert(ctx, repository.ProcedureDraft{
		ID: draftID, DocumentID: job.DocumentID, ProcedureID: job.ProcedureID,
		DraftDefinition: payload, ValidationResult: validationRaw,
		Status: repository.DraftStatusDraft, CreatedBy: &job.CreatedBy, UpdatedBy: &job.CreatedBy,
	})
	var pgErr *pgconn.PgError
	if errors.As(err, &pgErr) && pgErr.Code == "23505" {
		return draftID, nil
	}
	if err != nil {
		return uuid.Nil, err
	}
	return saved.ID, nil
}

func chunkIndexes(items []Excerpt) []int {
	out := make([]int, 0, len(items))
	for _, ex := range items {
		out = append(out, ex.ChunkIndex)
	}
	return out
}

func addUsage(total, part map[string]any) {
	for k, v := range part {
		n, ok := v.(float64)
		if !ok {
			continue
		}
		prev, _ := total[k].(float64)
		total[k] = prev + n
	}
}

// ResolveConflict records the Admin's choice for one source conflict: a
// candidate value (with its quotation as evidence) or -1 to keep the field as
// currently edited. The draft is revalidated; nothing is chosen automatically.
func (s *Service) ResolveConflict(ctx context.Context, actor, draftID uuid.UUID, req ResolveConflictRequest) (DraftView, error) {
	if req.ExpectedUpdatedAt == nil {
		return DraftView{}, fmt.Errorf("drafting: expected_updated_at is required")
	}
	if req.Choice == nil {
		return DraftView{}, fmt.Errorf("%w: choice is required", ErrReviewInvalid)
	}
	row, err := s.Drafts.Get(ctx, s.XAID, draftID)
	if err != nil {
		return DraftView{}, err
	}
	if row.Status != repository.DraftStatusDraft {
		return DraftView{}, fmt.Errorf("%w: only DRAFT rows can be changed", ErrReviewInvalid)
	}
	view, err := viewFromRow(row)
	if err != nil {
		return DraftView{}, err
	}
	idx := -1
	for i, c := range view.Conflicts {
		if c.ID == req.ConflictID {
			idx = i
		}
	}
	if idx < 0 {
		return DraftView{}, fmt.Errorf("%w: unknown conflict %s", ErrReviewInvalid, req.ConflictID)
	}
	conflict := &view.Conflicts[idx]
	choice := *req.Choice
	if choice < -1 || choice >= len(conflict.Candidates) {
		return DraftView{}, fmt.Errorf("%w: invalid choice", ErrReviewInvalid)
	}
	if choice >= 0 {
		cand := conflict.Candidates[choice]
		var obj map[string]any
		if err := json.Unmarshal(view.Definition, &obj); err != nil {
			return DraftView{}, err
		}
		if conflict.Field == "citations" {
			if len(cand.Citation) == 0 {
				return DraftView{}, fmt.Errorf("%w: candidate has no citation", ErrReviewInvalid)
			}
			var chosen any
			if err := json.Unmarshal(cand.Citation, &chosen); err != nil {
				return DraftView{}, err
			}
			current, _ := obj["citations"].([]any)
			at := -1
			for i, c := range current {
				raw, _ := json.Marshal(c)
				if citationConflictID(citationIdentity(raw)) == conflict.ID {
					at = i
				}
			}
			if at < 0 {
				return DraftView{}, fmt.Errorf("%w: this document is no longer cited in the Definition; keep the current value instead", ErrReviewInvalid)
			}
			current[at] = chosen
			obj["citations"] = current
		} else if key, isSlot := strings.CutPrefix(conflict.Field, "slots."); isSlot {
			var slot map[string]any
			if err := json.Unmarshal(cand.Slot, &slot); err != nil || slot == nil {
				return DraftView{}, fmt.Errorf("%w: candidate has no slot definition", ErrReviewInvalid)
			}
			slots, _ := obj["slots"].(map[string]any)
			if slots == nil {
				slots = map[string]any{}
			}
			slots[key] = slot
			obj["slots"] = slots
		} else {
			guidance, _ := obj["guidance"].(map[string]any)
			if guidance == nil {
				guidance = map[string]any{}
			}
			guidance[strings.TrimPrefix(conflict.Field, "guidance.")] = cand.Value
			obj["guidance"] = guidance
		}
		if view.Definition, err = json.Marshal(obj); err != nil {
			return DraftView{}, err
		}
		kept := view.Evidence[:0:0]
		for _, ev := range view.Evidence {
			if ev.ProposalID != conflict.ID {
				kept = append(kept, ev)
			}
		}
		if cand.Evidence != nil {
			ev := *cand.Evidence
			ev.ProposalID, ev.RawProposalID = conflict.ID, conflict.ID
			ev.BoundFingerprint, ev.ProposalFingerprint, ev.ReviewStatus, ev.Status = "", "", "", ""
			kept = append(kept, ev)
		}
		view.Evidence = kept
	}
	// Provenance follows the decision: the chosen candidate's quotations, or
	// none when the Admin kept a hand-edited value.
	setProvenance(&view, conflict.ID, choice, conflict.Candidates)
	now := time.Now().UTC()
	conflict.Status = ConflictResolved
	conflict.Chosen = &choice
	conflict.ResolvedBy = &actor
	conflict.ResolvedAt = &now
	conflict.ResolvedBasis = conflictValueBasis(view.Definition, *conflict)
	conflict.Reopened = false
	snap, err := s.resolveViewSource(ctx, view)
	if err != nil {
		return DraftView{}, err
	}
	view.Evidence = invalidateStaleEvidence(view.Evidence, view.Definition)
	revalidate(&view, snap, row.ProcedureCode, view.OriginalProcedureCode)
	payload, err := json.Marshal(storedPayloadFromView(view))
	if err != nil {
		return DraftView{}, err
	}
	validationRaw, err := json.Marshal(view.Validation)
	if err != nil {
		return DraftView{}, err
	}
	saved, err := s.Drafts.UpdateDefinition(ctx, s.XAID, draftID, actor, payload, validationRaw, *req.ExpectedUpdatedAt)
	if err != nil {
		if errors.Is(err, repository.ErrNotFound) {
			return DraftView{}, ErrConflict
		}
		return DraftView{}, err
	}
	return viewFromRow(saved)
}

// AcknowledgeGap records that a coverage-gap field stays empty on purpose (the
// source does not actually state it). A reason is required; filling the field
// from the source instead makes the gap disappear on its own.
func (s *Service) AcknowledgeGap(ctx context.Context, actor, draftID uuid.UUID, req AcknowledgeGapRequest) (DraftView, error) {
	if req.ExpectedUpdatedAt == nil {
		return DraftView{}, fmt.Errorf("drafting: expected_updated_at is required")
	}
	reason := strings.TrimSpace(req.Reason)
	if len([]rune(reason)) < 5 || len([]rune(reason)) > 500 {
		return DraftView{}, fmt.Errorf("%w: reason must be 5-500 characters", ErrReviewInvalid)
	}
	row, err := s.Drafts.Get(ctx, s.XAID, draftID)
	if err != nil {
		return DraftView{}, err
	}
	if row.Status != repository.DraftStatusDraft {
		return DraftView{}, fmt.Errorf("%w: only DRAFT rows can be changed", ErrReviewInvalid)
	}
	view, err := viewFromRow(row)
	if err != nil {
		return DraftView{}, err
	}
	if view.Coverage == nil {
		return DraftView{}, fmt.Errorf("%w: draft has no coverage report", ErrReviewInvalid)
	}
	snap, err := s.resolveViewSource(ctx, view)
	if err != nil {
		return DraftView{}, err
	}
	// Decide only on a finding that exists now, and pin the decision to its state.
	revalidate(&view, snap, row.ProcedureCode, view.OriginalProcedureCode)
	var target *ackFinding
	for _, f := range pendingFindings(view.Coverage) {
		if f.Field == req.Field {
			f := f
			target = &f
		}
	}
	if target == nil {
		return DraftView{}, fmt.Errorf("%w: %s is not a pending coverage finding", ErrReviewInvalid, req.Field)
	}
	acks := view.Coverage.Acknowledged[:0:0]
	for _, a := range view.Coverage.Acknowledged {
		if a.Field != req.Field {
			acks = append(acks, a)
		}
	}
	view.Coverage.Acknowledged = append(acks, GapAck{Field: target.Field, Basis: target.Basis, Reason: reason, By: actor, At: time.Now().UTC()})
	revalidate(&view, snap, row.ProcedureCode, view.OriginalProcedureCode)
	payload, err := json.Marshal(storedPayloadFromView(view))
	if err != nil {
		return DraftView{}, err
	}
	validationRaw, err := json.Marshal(view.Validation)
	if err != nil {
		return DraftView{}, err
	}
	saved, err := s.Drafts.UpdateDefinition(ctx, s.XAID, draftID, actor, payload, validationRaw, *req.ExpectedUpdatedAt)
	if err != nil {
		if errors.Is(err, repository.ErrNotFound) {
			return DraftView{}, ErrConflict
		}
		return DraftView{}, err
	}
	return viewFromRow(saved)
}

func setProvenance(view *DraftView, proposalID string, choice int, cands []ConflictCandidate) {
	kept := view.Provenance[:0:0]
	for _, p := range view.Provenance {
		if p.ProposalID != proposalID {
			kept = append(kept, p)
		}
	}
	view.Provenance = kept
	if choice < 0 {
		return
	}
	props, err := EnumerateMaterialProposals(view.Definition)
	if err != nil {
		return
	}
	for _, p := range props {
		if p.ID == proposalID {
			sources := append([]ProvenanceRef{}, cands[choice].Sources...)
			view.Provenance = append(view.Provenance, ProposalProvenance{ProposalID: p.ID, Content: p.Content, Sources: sources})
			return
		}
	}
}

// CitationHardMax bounds a Definition's citations when an Admin keeps extra
// ones; the draft validator enforces the same limit on every draft.
const CitationHardMax = decision.MaxDraftCitations

// KeepCitation puts a citation dropped by the merge limit back into the
// Definition, swapping out a current one or appending. The truncation list is
// updated (a swapped-out citation becomes the dropped one), so any earlier
// "reviewed" decision on it no longer matches and must be taken again.
func (s *Service) KeepCitation(ctx context.Context, actor, draftID uuid.UUID, req KeepCitationRequest) (DraftView, error) {
	if req.ExpectedUpdatedAt == nil {
		return DraftView{}, fmt.Errorf("drafting: expected_updated_at is required")
	}
	row, err := s.Drafts.Get(ctx, s.XAID, draftID)
	if err != nil {
		return DraftView{}, err
	}
	if row.Status != repository.DraftStatusDraft {
		return DraftView{}, fmt.Errorf("%w: only DRAFT rows can be changed", ErrReviewInvalid)
	}
	view, err := viewFromRow(row)
	if err != nil {
		return DraftView{}, err
	}
	var trunc *Truncation
	if view.Coverage != nil {
		for i := range view.Coverage.Truncated {
			if view.Coverage.Truncated[i].Field == "citations" {
				trunc = &view.Coverage.Truncated[i]
			}
		}
	}
	if trunc == nil || req.Dropped < 0 || req.Dropped >= len(trunc.DroppedCitations) {
		return DraftView{}, fmt.Errorf("%w: unknown dropped citation", ErrReviewInvalid)
	}
	var obj map[string]any
	if err := json.Unmarshal(view.Definition, &obj); err != nil {
		return DraftView{}, err
	}
	current, _ := obj["citations"].([]any)
	var keep any
	keepRaw := trunc.DroppedCitations[req.Dropped]
	if err := json.Unmarshal(keepRaw, &keep); err != nil {
		return DraftView{}, err
	}
	rest := append([]json.RawMessage{}, trunc.DroppedCitations[:req.Dropped]...)
	rest = append(rest, trunc.DroppedCitations[req.Dropped+1:]...)
	present := -1
	for i, c := range current {
		raw, _ := json.Marshal(c)
		if citationIdentity(raw) == citationIdentity(keepRaw) {
			present = i
		}
	}
	switch {
	case present >= 0 && req.Replace != nil:
		// Replacing would drop a valid citation for one that is already there.
		return DraftView{}, fmt.Errorf("%w: this source is already citation %d of the Definition; nothing to replace", ErrReviewInvalid, present+1)
	case present >= 0:
		// Already kept (e.g. added by hand in the JSON): just clear it from the
		// dropped list, never add it twice.
	case req.Replace != nil:
		r := *req.Replace
		if r < 0 || r >= len(current) {
			return DraftView{}, fmt.Errorf("%w: invalid citation to replace", ErrReviewInvalid)
		}
		out, _ := json.Marshal(current[r])
		rest = append(rest, out)
		current[r] = keep
	default:
		if len(current) >= CitationHardMax {
			return DraftView{}, fmt.Errorf("%w: at most %d citations; replace one instead", ErrReviewInvalid, CitationHardMax)
		}
		current = append(current, keep)
	}
	obj["citations"] = current
	if view.Definition, err = json.Marshal(obj); err != nil {
		return DraftView{}, err
	}
	trunc.DroppedCitations = rest
	trunc.Dropped = make([]string, 0, len(rest))
	for _, raw := range rest {
		trunc.Dropped = append(trunc.Dropped, citationLabel(raw))
	}
	kept := view.Coverage.Truncated[:0:0]
	for _, t := range view.Coverage.Truncated {
		if t.Field != "citations" || len(t.DroppedCitations) > 0 {
			kept = append(kept, t)
		}
	}
	view.Coverage.Truncated = kept
	snap, err := s.resolveViewSource(ctx, view)
	if err != nil {
		return DraftView{}, err
	}
	revalidate(&view, snap, row.ProcedureCode, view.OriginalProcedureCode)
	payload, err := json.Marshal(storedPayloadFromView(view))
	if err != nil {
		return DraftView{}, err
	}
	validationRaw, err := json.Marshal(view.Validation)
	if err != nil {
		return DraftView{}, err
	}
	saved, err := s.Drafts.UpdateDefinition(ctx, s.XAID, draftID, actor, payload, validationRaw, *req.ExpectedUpdatedAt)
	if err != nil {
		if errors.Is(err, repository.ErrNotFound) {
			return DraftView{}, ErrConflict
		}
		return DraftView{}, err
	}
	return viewFromRow(saved)
}
