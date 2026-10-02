package decision

import "testing"

func catalogCandidates() []IntentCandidate {
	return []IntentCandidate{
		{
			ProcedureCode: "dk_khai_sinh",
			Name:          "Đăng ký khai sinh",
			Examples: []string{
				"Tôi muốn làm giấy khai sinh cho con",
				"Đăng ký khai sinh",
				"Làm khai sinh cho bé mới sinh",
			},
		},
		{
			ProcedureCode: "chung_thuc_ban_sao",
			Name:          "Chứng thực bản sao",
			Examples: []string{
				"Chứng thực bản sao CCCD",
				"Tôi muốn chứng thực bằng đại học",
			},
		},
	}
}

func TestMatchIntentHighConfidence(t *testing.T) {
	m := MatchIntentBands("Tôi muốn đăng ký khai sinh cho con", catalogCandidates())
	if m.Band != IntentSelect {
		t.Fatalf("band=%s candidates=%v", m.Band, m.Candidates)
	}
	if m.Selected.ProcedureCode != "dk_khai_sinh" {
		t.Fatalf("selected=%s", m.Selected.ProcedureCode)
	}
}

func TestMatchIntentTypoStillMatchesOrConfirms(t *testing.T) {
	m := MatchIntentBands("dang ky khai sinhh", catalogCandidates())
	if m.Band == IntentNone {
		t.Fatalf("typo fell to none: %#v", m)
	}
	if m.Band == IntentSelect && m.Selected.ProcedureCode != "dk_khai_sinh" {
		t.Fatalf("wrong select: %#v", m)
	}
	if m.Band == IntentConfirm {
		found := false
		for _, c := range m.Candidates {
			if c.ProcedureCode == "dk_khai_sinh" {
				found = true
			}
		}
		if !found {
			t.Fatalf("confirm missing khai sinh: %#v", m.Candidates)
		}
	}
}

func TestMatchIntentLowConfidenceOutOfScope(t *testing.T) {
	m := MatchIntentBands("thời tiết hôm nay thế nào", catalogCandidates())
	if m.Band != IntentNone {
		t.Fatalf("expected none, got %#v", m)
	}
}

func TestMatchIntentNearTieConfirm(t *testing.T) {
	// Ambiguous short query that can hit both loosely — force mid band via generic phrase.
	cands := []IntentCandidate{
		{ProcedureCode: "a", Name: "Đăng ký khai sinh", Examples: []string{"đăng ký giấy tờ khai sinh"}},
		{ProcedureCode: "b", Name: "Đăng ký kết hôn", Examples: []string{"đăng ký giấy tờ kết hôn"}},
	}
	m := MatchIntentBands("đăng ký giấy tờ", cands)
	if m.Band == IntentSelect {
		// Acceptable if clear winner; otherwise must confirm.
		return
	}
	if m.Band != IntentConfirm {
		t.Fatalf("expected confirm or select, got %#v", m)
	}
	if len(m.Candidates) < 2 {
		t.Fatalf("expected multiple candidates, got %#v", m.Candidates)
	}
}

func TestMatchIntentBinaryHelper(t *testing.T) {
	got, ok := MatchIntent("Chứng thực bản sao CCCD", catalogCandidates())
	if !ok || got.ProcedureCode != "chung_thuc_ban_sao" {
		t.Fatalf("got=%#v ok=%v", got, ok)
	}
	_, ok = MatchIntent("xyz abc", catalogCandidates())
	if ok {
		t.Fatal("expected no match")
	}
}
