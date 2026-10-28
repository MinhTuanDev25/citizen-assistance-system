package drafting

import (
	"errors"
	"strings"
	"testing"
)

func TestCheckDraftSizeRejectsWholeDocumentRanges(t *testing.T) {
	snap := SourceSnapshot{SourcePageStart: 1, SourcePageEnd: 451}
	for i := 0; i < MaxDraftExcerpts; i++ {
		snap.Excerpts = append(snap.Excerpts, Excerpt{Content: "Điều 1. Hồ sơ gồm tờ khai."})
	}
	if err := checkDraftSize(snap); err != nil {
		t.Fatalf("exactly the limit must pass: %v", err)
	}
	snap.Excerpts = append(snap.Excerpts, Excerpt{Content: "một đoạn nữa"})
	err := checkDraftSize(snap)
	if !errors.Is(err, ErrSourceTooLarge) || !strings.Contains(err.Error(), "pages 1-451 have 81 excerpts") {
		t.Fatalf("over the limit: %v", err)
	}
	big := SourceSnapshot{SourcePageStart: 10, SourcePageEnd: 12, Excerpts: []Excerpt{{Content: strings.Repeat("a", MaxDraftContentBytes+1)}}}
	if !errors.Is(checkDraftSize(big), ErrSourceTooLarge) {
		t.Fatal("content over the byte budget must be rejected")
	}
}

func TestReviewWithoutEvidenceIsAValidationError(t *testing.T) {
	view := DraftView{Evidence: []EvidenceItem{{ProposalID: "slot:a", QuoteValid: true, ReviewStatus: ReviewPending}}}
	for _, d := range []ReviewDecision{
		{ProposalID: "slot:yeu_cau_ban_sao", Decision: ReviewRejected},
		{ProposalID: "slot:a", Decision: "maybe"},
	} {
		if err := applyReviewDecisions(&view, []ReviewDecision{d}); !errors.Is(err, ErrReviewInvalid) {
			t.Fatalf("%+v: want ErrReviewInvalid, got %v", d, err)
		}
	}
	if err := applyReviewDecisions(&view, []ReviewDecision{{ProposalID: "slot:a", Decision: ReviewAccepted}}); err != nil {
		t.Fatal(err)
	}
}
