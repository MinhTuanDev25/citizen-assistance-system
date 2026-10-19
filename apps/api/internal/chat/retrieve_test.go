package chat

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"strings"
	"testing"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/decision"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/repository"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/retrieve"
	"github.com/google/uuid"
)

func TestRetrievalQueryUsesDefinitionNotSlotFill(t *testing.T) {
	slot := "1. Bệnh viện Đa khoa tỉnh\n2. rồi\n3. có"
	plan := &planned{
		procedureName: "Đăng ký khai sinh",
		result: TurnResult{
			ProcedureCode: "dk_khai_sinh",
			Guidance: &decision.Guidance{
				Summary:   "Sau khi đủ thông tin, chuẩn bị hồ sơ đăng ký khai sinh.",
				Checklist: []string{"Bản chính giấy chứng sinh do cơ sở y tế cấp", "Tờ khai đăng ký khai sinh"},
			},
		},
	}
	q := retrievalQuery(slot, plan)
	if q == slot || strings.Contains(q, "Bệnh viện") || strings.Contains(q, "rồi") {
		t.Fatalf("slot-fill utterance must not be the E5 query, got %q", q)
	}
	if !strings.Contains(q, "Đăng ký khai sinh") {
		t.Fatalf("procedure name missing: %q", q)
	}
	if !strings.Contains(q, "giấy chứng sinh") {
		t.Fatalf("checklist missing: %q", q)
	}
	if !strings.Contains(q, "dk_khai_sinh") {
		t.Fatalf("procedure code missing: %q", q)
	}
}

func TestRetrievalQueryFallsBackToMessageWithoutDefinition(t *testing.T) {
	q := retrievalQuery("  Tôi muốn đăng ký khai sinh  ", &planned{})
	if q != "Tôi muốn đăng ký khai sinh" {
		t.Fatalf("%q", q)
	}
}

func TestRetrievalQueryClipsLongDefinition(t *testing.T) {
	plan := &planned{
		result: TurnResult{
			Guidance: &decision.Guidance{Summary: strings.Repeat("khai sinh ", 400)},
		},
	}
	q := retrievalQuery("ignored", plan)
	if got := len([]rune(q)); got != liveQueryMaxRunes {
		t.Fatalf("len=%d", got)
	}
}

func TestAttachHTTPSourcesRequestsSearchK(t *testing.T) {
	ver := uuid.New()
	proc := uuid.New()
	stub := &stubRetriever{resp: &retrieve.Response{Status: "no_relevant_evidence"}}
	plan := &planned{
		result: TurnResult{Action: "PROVIDE_FINAL_GUIDANCE"},
		save:   repository.SaveTurnInput{ProcedureID: &proc, VersionID: &ver, RequestID: uuid.New()},
	}
	attachHTTPSources(nil, stub, uuid.Nil, "xa_chu_se", "khai sinh", nil, []repository.SourcePin{{
		DocumentID: uuid.New(), GenerationID: uuid.New(),
	}}, plan)
	if stub.got.TopK != liveSearchK {
		t.Fatalf("top_k=%d want search_k=%d", stub.got.TopK, liveSearchK)
	}
	if liveSearchK <= liveDisplayK || liveSearchK > retrieve.MaxTopK {
		t.Fatalf("search_k=%d display_k=%d max=%d", liveSearchK, liveDisplayK, retrieve.MaxTopK)
	}
}

func TestAttachHTTPSourcesCapsDisplayK(t *testing.T) {
	ver := uuid.MustParse("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
	doc := uuid.MustParse("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
	gen := uuid.MustParse("cccccccc-cccc-cccc-cccc-cccccccccccc")
	score := 0.81
	hits := make([]retrieve.Hit, liveDisplayK+4)
	for i := range hits {
		hits[i] = retrieve.Hit{
			ChunkID:            uuid.NewString(),
			DocumentID:         doc.String(),
			GenerationID:       gen.String(),
			ProcedureVersionID: ver.String(),
			Excerpt:            "Bản chính giấy chứng sinh do cơ sở y tế cấp",
			EvidenceStatus:     "related",
			PageStart:          i + 1,
			PageEnd:            i + 1,
			RetrievalScore:     &score,
		}
	}
	stub := &stubRetriever{resp: &retrieve.Response{Status: "ok", Hits: hits}}
	plan := &planned{
		result: TurnResult{Action: "PROVIDE_FINAL_GUIDANCE"},
		save:   repository.SaveTurnInput{ProcedureID: &doc, VersionID: &ver, RequestID: uuid.New()},
	}
	attachHTTPSources(nil, stub, uuid.Nil, "xa_chu_se", "khai sinh", []string{"Bản chính giấy chứng sinh do cơ sở y tế cấp"}, []repository.SourcePin{{
		DocumentID: doc, GenerationID: gen,
	}}, plan)
	if plan.result.RetrievalStatus != "ok" {
		t.Fatalf("status=%s", plan.result.RetrievalStatus)
	}
	if len(plan.result.Citations) != liveDisplayK {
		t.Fatalf("cites=%d want display_k=%d", len(plan.result.Citations), liveDisplayK)
	}
	for _, c := range plan.result.Citations {
		if c.EvidenceStatus != "related" {
			t.Fatalf("must stay related, got %s", c.EvidenceStatus)
		}
	}
}

func TestAttachHTTPSourcesG3IdentityStaysRelatedWithoutAllowlist(t *testing.T) {
	ver := uuid.MustParse("b27d95a8-6a24-4bae-8631-2b6a14e8a41c")
	doc := uuid.MustParse("60b677df-acc0-4198-a3b6-a919c01482a1")
	gen := uuid.MustParse("e2ad49da-4db7-4e18-90e3-c2f0bb358940")
	score := 0.81
	related := retrieve.Hit{
		ChunkID: "11111111-1111-1111-1111-111111111111", DocumentID: doc.String(), GenerationID: gen.String(),
		ProcedureVersionID: ver.String(), Excerpt: "Bản chính giấy chứng sinh do cơ sở y tế cấp",
		PageStart: 11, PageEnd: 11, RetrievalScore: &score, ContentSHA256: "not-a-mapped-hash",
	}
	former := retrieve.Hit{
		ChunkID: "22222222-2222-2222-2222-222222222222", DocumentID: doc.String(), GenerationID: gen.String(),
		ProcedureVersionID: ver.String(), Excerpt: "Giấy tờ phải nộp:-Giấy chứng sinh;trường hợp không có",
		PageStart: 12, PageEnd: 12, RetrievalScore: &score,
		ContentSHA256: "410276e9993151a4f2ff17e5e8a3ea7b5cdc6abd42faf6ded0ac07cf57e8b17c",
	}
	hits := []retrieve.Hit{related, related, related, former}
	stub := &stubRetriever{resp: &retrieve.Response{Status: "ok", Hits: hits}}
	stmt := "Giấy chứng sinh (nếu có) hoặc giấy tờ thay thế theo hướng dẫn xã"
	plan := &planned{
		result: TurnResult{Action: "PROVIDE_FINAL_GUIDANCE", ProcedureCode: "dk_khai_sinh"},
		save:   repository.SaveTurnInput{ProcedureID: &doc, VersionID: &ver, RequestID: uuid.New()},
	}
	attachHTTPSources(nil, stub, uuid.Nil, "xa_chu_se", stmt, []string{stmt}, []repository.SourcePin{{
		DocumentID: doc, GenerationID: gen,
	}}, plan)
	if len(plan.result.Citations) != liveDisplayK {
		t.Fatalf("cites=%d", len(plan.result.Citations))
	}
	for _, c := range plan.result.Citations {
		if c.EvidenceStatus != "related" {
			t.Fatalf("demoted allowlist must keep related, got %#v", plan.result.Citations)
		}
		if c.GenerationID != gen.String() || c.DocumentID != doc.String() {
			t.Fatalf("pin identity drifted: %#v", c)
		}
	}
}

func TestPickPassagesRanksOverlapThenKeepsLinkedFile(t *testing.T) {
	rows := []sourceChunk{
		{Title: "Kiến trúc", Content: "software architecture overview"},
		{Title: "Chứng thực", Content: "Mang bản chính giấy tờ đến UBND xã để chứng thực bản sao"},
		{Title: "Phụ lục", Content: "trang phụ không liên quan"},
	}
	got := pickPassages("Tôi muốn chứng thực bản sao", rows, 2)
	if len(got) != 1 {
		t.Fatalf("len=%d %#v", len(got), got)
	}
	if got[0].Title != "Chứng thực" {
		t.Fatalf("want chứng thực first, got %#v", got[0].Title)
	}
}

func TestPickPassagesDropsChunksWithoutOverlap(t *testing.T) {
	rows := []sourceChunk{
		{Title: "hi", Content: "Citizen Assistance System architecture"},
	}
	got := pickPassages("khai sinh", rows, 3)
	if len(got) != 0 {
		t.Fatalf("unrelated chunk must not be evidence, got %#v", got)
	}
}

func TestAttachLiveSourcesSkipsAskTurns(t *testing.T) {
	plan := &planned{result: TurnResult{Action: "ASK_MISSING_SLOTS", ReplyText: "hỏi slot"}}
	attachLiveSources(nil, nil, nil, nil, uuid.Nil, "xa_chu_se", "chứng thực", plan)
	if plan.result.ReplyText != "hỏi slot" {
		t.Fatal("ask turns must not grow a PDF appendix")
	}
}

func TestSnippetTrims(t *testing.T) {
	s := snippet("  a   b  ", 10)
	if s != "a b" {
		t.Fatalf("%q", s)
	}
	long := strings.Repeat("n", 40)
	if !strings.HasSuffix(snippet(long, 10), "…") {
		t.Fatal("expected ellipsis")
	}
}

func TestAttachLiveSourcesNilPlan(t *testing.T) {
	attachLiveSources(nil, nil, nil, nil, uuid.Nil, "xa", "x", nil)
}

func TestClassifySupportNeverTreatsOverlapAsProof(t *testing.T) {
	overlap := retrieve.Hit{Excerpt: "đăng ký khai sinh tại xã"}
	if classifySupport(overlap, nil, "dk_khai_sinh") != "related" {
		t.Fatal("no statement must stay related")
	}
	if classifySupport(overlap, []string{"Mang bản chính giấy chứng sinh"}, "dk_khai_sinh") != "related" {
		t.Fatal("topic overlap is not supporting evidence")
	}
	item := "Bản chính giấy chứng sinh do cơ sở y tế cấp"
	if classifySupport(retrieve.Hit{Excerpt: item + " nộp tại UBND xã"}, []string{item}, "dk_khai_sinh") != "related" {
		t.Fatal("substring overlap is not supporting evidence")
	}
	if classifySupport(retrieve.Hit{Excerpt: "Không cần nộp " + item}, []string{item}, "dk_khai_sinh") != "related" {
		t.Fatal("negation must not be labeled supporting")
	}
}

func TestHitsToCitationsHashesTruncatedExcerpt(t *testing.T) {
	long := strings.Repeat("nội dung giấy chứng sinh ", 40)
	hits := []repository.PinnedChunk{{
		Content:      long,
		Title:        "HD",
		DocumentID:   uuid.New(),
		GenerationID: uuid.New(),
		ChunkID:      uuid.New(),
		PageStart:    2,
		PageEnd:      2,
	}}
	cites, saved := hitsToCitations(hits, uuid.New(), nil, "dk_khai_sinh")
	if len(cites) != 1 || cites[0].EvidenceStatus != "related" {
		t.Fatalf("%#v", cites)
	}
	want := snippet(long, liveSnippetRunes)
	if cites[0].Excerpt != want {
		t.Fatalf("excerpt %q", cites[0].Excerpt)
	}
	sum := sha256.Sum256([]byte(want))
	if cites[0].ExcerptSHA256 != hex.EncodeToString(sum[:]) {
		t.Fatal("hash must cover the persisted excerpt")
	}
	if saved[0]["excerpt_sha256"] != cites[0].ExcerptSHA256 {
		t.Fatal("save row hash drifted from citation")
	}
}

func TestPinSnapshotTitlePrefersFrozenMetadata(t *testing.T) {
	doc := uuid.New()
	title, number := pinSnapshotTitle("live title", "live-no", doc.String(), []repository.SourcePin{{
		DocumentID:     doc,
		Title:          "Pinned title",
		DocumentNumber: "01/PIN",
	}})
	if title != "Pinned title" || number != "01/PIN" {
		t.Fatalf("%s %s", title, number)
	}
}

type stubRetriever struct {
	resp *retrieve.Response
	err  error
	got  retrieve.Request
}

func (s *stubRetriever) Retrieve(_ context.Context, req retrieve.Request) (*retrieve.Response, error) {
	s.got = req
	return s.resp, s.err
}

func TestAttachHTTPSourcesRejectsHitsOutsidePins(t *testing.T) {
	ver := uuid.MustParse("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
	doc := uuid.MustParse("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
	gen := uuid.MustParse("cccccccc-cccc-cccc-cccc-cccccccccccc")
	stub := &stubRetriever{
		resp: &retrieve.Response{
			Status: "ok",
			Hits: []retrieve.Hit{{
				ChunkID:            uuid.NewString(),
				DocumentID:         uuid.NewString(),
				GenerationID:       uuid.NewString(),
				ProcedureVersionID: ver.String(),
				Title:              "wrong generation",
				Excerpt:            "Mang bản chính giấy chứng sinh",
				EvidenceStatus:     "supporting",
				PageStart:          3,
				PageEnd:            3,
			}},
		},
	}
	plan := &planned{
		result: TurnResult{
			Action:   "PROVIDE_FINAL_GUIDANCE",
			Guidance: &decision.Guidance{Checklist: []string{"Mang bản chính giấy chứng sinh"}},
		},
		save: repository.SaveTurnInput{ProcedureID: &doc, VersionID: &ver},
	}
	attachHTTPSources(nil, stub, uuid.Nil, "xa_chu_se", "khai sinh", plan.result.Guidance.Checklist, []repository.SourcePin{{
		DocumentID: doc, GenerationID: gen,
	}}, plan)
	if plan.result.RetrievalStatus != "no_relevant_evidence" {
		t.Fatalf("out-of-scope hit must be dropped, got %q", plan.result.RetrievalStatus)
	}
	if len(plan.result.Citations) != 0 {
		t.Fatal("must not persist a citation from another generation")
	}
}

func TestAttachHTTPSourcesUnavailableDoesNotInventCitations(t *testing.T) {
	ver := uuid.New()
	proc := uuid.New()
	stub := &stubRetriever{err: retrieve.ErrUnavailable}
	plan := &planned{
		result: TurnResult{Action: "DIRECT_ANSWER", ReplyText: "hướng dẫn từ definition"},
		save:   repository.SaveTurnInput{ProcedureID: &proc, VersionID: &ver},
	}
	attachHTTPSources(nil, stub, uuid.Nil, "xa_chu_se", "khai sinh", nil, []repository.SourcePin{{
		DocumentID: uuid.New(), GenerationID: uuid.New(),
	}}, plan)
	if plan.result.RetrievalStatus != "retrieval_unavailable" {
		t.Fatalf("got %q", plan.result.RetrievalStatus)
	}
	if plan.result.RetrievalReason != "" {
		t.Fatalf("plain ErrUnavailable must not invent a reason, got %q", plan.result.RetrievalReason)
	}
	if plan.result.ReplyText != "hướng dẫn từ definition" {
		t.Fatal("definition reply must stay")
	}
	if len(plan.result.Citations) != 0 {
		t.Fatal("service error must not create citations")
	}
}

func TestAttachHTTPSourcesKeepsPinnedHit(t *testing.T) {
	ver := uuid.MustParse("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
	doc := uuid.MustParse("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
	gen := uuid.MustParse("cccccccc-cccc-cccc-cccc-cccccccccccc")
	score := 0.81
	stub := &stubRetriever{
		resp: &retrieve.Response{
			Status: "ok",
			Hits: []retrieve.Hit{{
				ChunkID:            "dddddddd-dddd-dddd-dddd-dddddddddddd",
				DocumentID:         doc.String(),
				GenerationID:       gen.String(),
				ProcedureVersionID: ver.String(),
				Title:              "Hướng dẫn khai sinh",
				Excerpt:            "Bản chính giấy chứng sinh do cơ sở y tế cấp",
				EvidenceStatus:     "supporting",
				PageStart:          2,
				PageEnd:            2,
				RetrievalScore:     &score,
				Availability:       "available",
			}},
		},
	}
	plan := &planned{
		result: TurnResult{Action: "PROVIDE_FINAL_GUIDANCE"},
		save:   repository.SaveTurnInput{ProcedureID: &doc, VersionID: &ver, RequestID: uuid.New()},
	}
	attachHTTPSources(nil, stub, uuid.Nil, "xa_chu_se", "khai sinh", []string{"Bản chính giấy chứng sinh do cơ sở y tế cấp"}, []repository.SourcePin{{
		DocumentID: doc, GenerationID: gen, Title: "Pinned HD", DocumentNumber: "01/PIN",
	}}, plan)
	if plan.result.RetrievalStatus != "ok" || len(plan.result.Citations) != 1 {
		t.Fatalf("status=%s cites=%d", plan.result.RetrievalStatus, len(plan.result.Citations))
	}
	if plan.result.Citations[0].Title != "Pinned HD" {
		t.Fatalf("title=%s", plan.result.Citations[0].Title)
	}
	if plan.result.Citations[0].EvidenceStatus != "related" {
		t.Fatalf("status=%s", plan.result.Citations[0].EvidenceStatus)
	}
	wantExcerpt := snippet("Bản chính giấy chứng sinh do cơ sở y tế cấp", liveSnippetRunes)
	sum := sha256.Sum256([]byte(wantExcerpt))
	if plan.result.Citations[0].ExcerptSHA256 != hex.EncodeToString(sum[:]) {
		t.Fatal("HTTP path must hash the persisted excerpt, not the untruncated payload")
	}
	if stub.got.XaID != "xa_chu_se" || stub.got.ProcedureVersionID != ver.String() {
		t.Fatalf("scope %#v", stub.got)
	}
}

func TestMapRetrieveStatusKeepsSystemErrorsDistinct(t *testing.T) {
	cases := []struct {
		status, reason, want string
	}{
		{"ok", "", "ok"},
		{"no_relevant_evidence", "no_candidates", "no_relevant_evidence"},
		{"all_candidates_rejected", "postgres_mismatch", "all_candidates_rejected"},
		{"retrieval_incompatible", "embedding_identity_mismatch", "retrieval_incompatible"},
		{"retrieval_unavailable", "qdrant_unavailable", "retrieval_unavailable"},
		{"all_candidates_rejected", "embedding_identity_mismatch", "all_candidates_rejected"},
		{"mystery", "embedding_identity_mismatch", "retrieval_incompatible"},
		{"mystery", "", "retrieval_unavailable"},
	}
	for _, tc := range cases {
		if got := mapRetrieveStatus(tc.status, tc.reason); got != tc.want {
			t.Fatalf("%s/%s got %s want %s", tc.status, tc.reason, got, tc.want)
		}
	}
}

func TestAttachHTTPSourcesPreservesIncompatibleWithoutCitations(t *testing.T) {
	ver := uuid.New()
	proc := uuid.New()
	stub := &stubRetriever{resp: &retrieve.Response{
		Status: "retrieval_incompatible",
		Reason: "embedding_identity_mismatch",
		Hits:   []retrieve.Hit{},
	}}
	plan := &planned{
		result: TurnResult{Action: "DIRECT_ANSWER", ReplyText: "hướng dẫn từ definition"},
		save:   repository.SaveTurnInput{ProcedureID: &proc, VersionID: &ver},
	}
	attachHTTPSources(nil, stub, uuid.Nil, "xa_chu_se", "khai sinh", nil, []repository.SourcePin{{
		DocumentID: uuid.New(), GenerationID: uuid.New(),
	}}, plan)
	if plan.result.RetrievalStatus != "retrieval_incompatible" {
		t.Fatalf("got %q", plan.result.RetrievalStatus)
	}
	if plan.result.RetrievalReason != "embedding_identity_mismatch" {
		t.Fatalf("reason %q", plan.result.RetrievalReason)
	}
	if plan.result.ReplyText != "hướng dẫn từ definition" {
		t.Fatal("definition reply must stay")
	}
	if len(plan.result.Citations) != 0 {
		t.Fatal("incompatible retrieve must not invent citations")
	}
}

func TestAttachHTTPSourcesCapsAllowedSources(t *testing.T) {
	ver := uuid.New()
	proc := uuid.New()
	pins := make([]repository.SourcePin, retrieve.MaxAllowedSources+7)
	for i := range pins {
		pins[i] = repository.SourcePin{DocumentID: uuid.New(), GenerationID: uuid.New()}
	}
	stub := &stubRetriever{resp: &retrieve.Response{Status: "no_relevant_evidence", Hits: []retrieve.Hit{}}}
	plan := &planned{
		result: TurnResult{Action: "PROVIDE_FINAL_GUIDANCE", ReplyText: "hướng dẫn"},
		save:   repository.SaveTurnInput{ProcedureID: &proc, VersionID: &ver, RequestID: uuid.New()},
	}
	attachHTTPSources(nil, stub, uuid.Nil, "xa_chu_se", "khai sinh", nil, pins, plan)
	if len(stub.got.AllowedSources) != retrieve.MaxAllowedSources {
		t.Fatalf("capped=%d want %d", len(stub.got.AllowedSources), retrieve.MaxAllowedSources)
	}
	if plan.result.RetrievalStatus != "no_relevant_evidence" {
		t.Fatalf("got %q", plan.result.RetrievalStatus)
	}
	if plan.result.RetrievalReason != "allowed_sources_truncated" {
		t.Fatalf("truncated retrieve must report partial coverage, got %q", plan.result.RetrievalReason)
	}
	if len(plan.result.Citations) != 0 {
		t.Fatal("capped retrieve must not invent citations")
	}
}

func TestAttachHTTPSourcesTruncatedOkKeepsHitsAndReason(t *testing.T) {
	ver := uuid.MustParse("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
	doc := uuid.MustParse("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
	gen := uuid.MustParse("cccccccc-cccc-cccc-cccc-cccccccccccc")
	score := 0.81
	pins := make([]repository.SourcePin, retrieve.MaxAllowedSources+1)
	pins[0] = repository.SourcePin{DocumentID: doc, GenerationID: gen, Title: "Pinned HD"}
	for i := 1; i < len(pins); i++ {
		pins[i] = repository.SourcePin{DocumentID: uuid.New(), GenerationID: uuid.New()}
	}
	stub := &stubRetriever{resp: &retrieve.Response{
		Status: "ok",
		Hits: []retrieve.Hit{{
			ChunkID:            uuid.NewString(),
			DocumentID:         doc.String(),
			GenerationID:       gen.String(),
			ProcedureVersionID: ver.String(),
			Excerpt:            "Bản chính giấy chứng sinh",
			EvidenceStatus:     "related",
			PageStart:          2,
			PageEnd:            2,
			RetrievalScore:     &score,
		}},
	}}
	plan := &planned{
		result: TurnResult{Action: "PROVIDE_FINAL_GUIDANCE"},
		save:   repository.SaveTurnInput{ProcedureID: &doc, VersionID: &ver, RequestID: uuid.New()},
	}
	attachHTTPSources(nil, stub, uuid.Nil, "xa_chu_se", "khai sinh", nil, pins, plan)
	if len(stub.got.AllowedSources) != retrieve.MaxAllowedSources {
		t.Fatalf("capped=%d", len(stub.got.AllowedSources))
	}
	if plan.result.RetrievalStatus != "ok" || plan.result.RetrievalReason != "allowed_sources_truncated" {
		t.Fatalf("status=%s reason=%s", plan.result.RetrievalStatus, plan.result.RetrievalReason)
	}
	if len(plan.result.Citations) != 1 {
		t.Fatalf("cites=%d", len(plan.result.Citations))
	}
}

func TestAttachHTTPSourcesPreservesUnavailableReason(t *testing.T) {
	ver := uuid.New()
	proc := uuid.New()
	stub := &stubRetriever{err: &retrieve.HTTPError{Status: 503, Reason: "embedding_unavailable"}}
	plan := &planned{
		result: TurnResult{Action: "DIRECT_ANSWER", ReplyText: "hướng dẫn từ definition"},
		save:   repository.SaveTurnInput{ProcedureID: &proc, VersionID: &ver},
	}
	attachHTTPSources(nil, stub, uuid.Nil, "xa_chu_se", "khai sinh", nil, []repository.SourcePin{{
		DocumentID: uuid.New(), GenerationID: uuid.New(),
	}}, plan)
	if plan.result.RetrievalStatus != "retrieval_unavailable" {
		t.Fatalf("got %q", plan.result.RetrievalStatus)
	}
	if plan.result.RetrievalReason != "embedding_unavailable" {
		t.Fatalf("reason %q", plan.result.RetrievalReason)
	}
	if len(plan.result.Citations) != 0 {
		t.Fatal("must not invent citations")
	}
}

func TestRelatednessTermsPreferChecklistOverUtterance(t *testing.T) {
	printer := "Tôi muốn xem thông tin sửa chữa máy in văn phòng"
	if overlap(tokens("Tôi muốn đăng ký khai sinh"), printer) <= 0 {
		t.Fatal("shared Tôi muốn must still match raw question overlap")
	}
	terms := relatednessTerms("Tôi muốn đăng ký khai sinh", []string{"Bản chính giấy chứng sinh do cơ sở y tế cấp"})
	if overlap(terms, printer) > 0 {
		t.Fatal("checklist overlap must not treat conversational printer text as related")
	}
	if overlap(terms, "Bản chính giấy chứng sinh nộp tại xã") <= 0 {
		t.Fatal("checklist tokens must keep the matching PDF excerpt")
	}
	if overlap(relatednessTerms("Dạ có", []string{"Bản chính giấy chứng sinh do cơ sở y tế cấp"}), "Bản chính giấy chứng sinh") <= 0 {
		t.Fatal("short slot answers must not drop checklist-overlapping excerpts")
	}
}

func TestAttachHTTPSourcesDropsPrinterOverlap(t *testing.T) {
	ver := uuid.MustParse("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
	doc := uuid.MustParse("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
	gen := uuid.MustParse("cccccccc-cccc-cccc-cccc-cccccccccccc")
	score := 0.11
	stub := &stubRetriever{resp: &retrieve.Response{
		Status: "ok",
		Hits: []retrieve.Hit{{
			ChunkID:            uuid.NewString(),
			DocumentID:         doc.String(),
			GenerationID:       gen.String(),
			ProcedureVersionID: ver.String(),
			Excerpt:            "Tôi muốn xem thông tin sửa chữa máy in văn phòng",
			PageStart:          3,
			PageEnd:            3,
			RetrievalScore:     &score,
		}},
	}}
	plan := &planned{
		result: TurnResult{Action: "PROVIDE_FINAL_GUIDANCE"},
		save:   repository.SaveTurnInput{ProcedureID: &doc, VersionID: &ver},
	}
	attachHTTPSources(nil, stub, uuid.Nil, "xa_chu_se", "Tôi muốn đăng ký khai sinh", []string{"Bản chính giấy chứng sinh do cơ sở y tế cấp"}, []repository.SourcePin{{
		DocumentID: doc, GenerationID: gen,
	}}, plan)
	if plan.result.RetrievalStatus != "no_relevant_evidence" {
		t.Fatalf("got %q", plan.result.RetrievalStatus)
	}
	if len(plan.result.Citations) != 0 {
		t.Fatal("printer-office text must not become related")
	}
}

func TestAttachHTTPSourcesDropsNullAndInvertedPages(t *testing.T) {
	ver := uuid.MustParse("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
	doc := uuid.MustParse("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
	gen := uuid.MustParse("cccccccc-cccc-cccc-cccc-cccccccccccc")
	score := 0.77
	stub := &stubRetriever{resp: &retrieve.Response{
		Status: "ok",
		Hits: []retrieve.Hit{
			{
				ChunkID: uuid.NewString(), DocumentID: doc.String(), GenerationID: gen.String(),
				ProcedureVersionID: ver.String(), Excerpt: "đăng ký khai sinh", RetrievalScore: &score,
			},
			{
				ChunkID: uuid.NewString(), DocumentID: doc.String(), GenerationID: gen.String(),
				ProcedureVersionID: ver.String(), Excerpt: "đăng ký khai sinh", PageStart: 12, PageEnd: 3, RetrievalScore: &score,
			},
		},
	}}
	plan := &planned{
		result: TurnResult{Action: "PROVIDE_FINAL_GUIDANCE"},
		save:   repository.SaveTurnInput{ProcedureID: &doc, VersionID: &ver},
	}
	attachHTTPSources(nil, stub, uuid.Nil, "xa_chu_se", "khai sinh", nil, []repository.SourcePin{{
		DocumentID: doc, GenerationID: gen,
	}}, plan)
	if plan.result.RetrievalStatus != "no_relevant_evidence" {
		t.Fatalf("got %q", plan.result.RetrievalStatus)
	}
	if len(plan.result.Citations) != 0 {
		t.Fatal("null or inverted pages must not become related")
	}
}

func TestAttachHTTPSourcesDropsBelowRelevanceFloor(t *testing.T) {
	ver := uuid.MustParse("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
	doc := uuid.MustParse("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
	gen := uuid.MustParse("cccccccc-cccc-cccc-cccc-cccccccccccc")
	low := 0.01
	stub := &stubRetriever{resp: &retrieve.Response{
		Status: "ok",
		Hits: []retrieve.Hit{{
			ChunkID:            uuid.NewString(),
			DocumentID:         doc.String(),
			GenerationID:       gen.String(),
			ProcedureVersionID: ver.String(),
			Excerpt:            "noise",
			RetrievalScore:     &low,
		}},
	}}
	plan := &planned{
		result: TurnResult{Action: "PROVIDE_FINAL_GUIDANCE"},
		save:   repository.SaveTurnInput{ProcedureID: &doc, VersionID: &ver},
	}
	attachHTTPSources(nil, stub, uuid.Nil, "xa_chu_se", "khai sinh", nil, []repository.SourcePin{{
		DocumentID: doc, GenerationID: gen,
	}}, plan)
	if plan.result.RetrievalStatus != "no_relevant_evidence" {
		t.Fatalf("got %q", plan.result.RetrievalStatus)
	}
	if len(plan.result.Citations) != 0 {
		t.Fatal("near-zero cosine must not become related")
	}
}

func TestAttachHTTPSourcesRejectedIsNotNoEvidence(t *testing.T) {
	ver := uuid.New()
	proc := uuid.New()
	stub := &stubRetriever{resp: &retrieve.Response{
		Status: "all_candidates_rejected",
		Reason: "postgres_mismatch",
	}}
	plan := &planned{
		result: TurnResult{Action: "PROVIDE_FINAL_GUIDANCE", ReplyText: "hướng dẫn"},
		save:   repository.SaveTurnInput{ProcedureID: &proc, VersionID: &ver},
	}
	attachHTTPSources(nil, stub, uuid.Nil, "xa", "x", nil, nil, plan)
	if plan.result.RetrievalStatus != "all_candidates_rejected" {
		t.Fatalf("got %q", plan.result.RetrievalStatus)
	}
	if len(plan.result.Citations) != 0 {
		t.Fatal("rejected candidates must not become citations")
	}
}
