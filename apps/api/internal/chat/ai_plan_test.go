package chat

import (
	"testing"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/aiextract"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/decision"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/repository"
	"github.com/google/uuid"
)

// These are pure Go-level unit tests (no DB, no HTTP) exercising the
// AI-assisted planning functions directly with a pre-validated
// aiextract.ValidatedProposal, exactly as Phase 3 of turnWithAI would
// construct it after aiextract.Validate has already run.

func TestPlanTurnAI_FreshSelectionFillsSlotButStillAsksMissing(t *testing.T) {
	snap := &repository.TurnSnapshot{
		Session: repository.Session{ID: uuid.New(), XaID: "xa_chu_se", Status: "OPEN"},
		Catalog: testCatalog(),
	}
	validated := &aiextract.ValidatedProposal{
		IntentProcedureCode: "dk_khai_sinh",
		IntentConfidence:    0.9,
		SlotsProcedureCode:  "dk_khai_sinh",
		Slots: map[string]aiextract.ValidatedSlot{
			"da_ket_hon": {Value: false, Operation: "set", Confidence: 0.9},
		},
	}
	meta := &aiMeta{}
	plan, err := planTurnAI(snap, "Tôi muốn đăng ký khai sinh", validated, decision.DefaultPolicy, meta)
	if err != nil {
		t.Fatalf("planTurnAI: %v", err)
	}
	if !plan.save.PinProcedure || plan.result.ProcedureCode != "dk_khai_sinh" {
		t.Fatalf("expected pin to dk_khai_sinh, got %+v", plan.result)
	}
	if v, ok := plan.result.SlotState["da_ket_hon"]; !ok || v.Value != false {
		t.Fatalf("expected da_ket_hon=false seeded, got %+v", plan.result.SlotState)
	}
	// noi_sinh and co_giay_chung_sinh are still missing — AI partial fill
	// must never let the turn skip past ASK_MISSING_SLOTS.
	if plan.result.Action != decision.ActionAsk {
		t.Fatalf("Action = %s, want %s", plan.result.Action, decision.ActionAsk)
	}
	if meta.source != "ai" {
		t.Fatalf("meta.source = %q, want ai", meta.source)
	}
	if len(meta.accepted) != 1 || meta.accepted[0] != "da_ket_hon" {
		t.Fatalf("meta.accepted = %v, want [da_ket_hon]", meta.accepted)
	}
}

func TestPlanTurnAI_LowConfidenceIntentFallsBackToKeyword(t *testing.T) {
	snap := &repository.TurnSnapshot{
		Session: repository.Session{ID: uuid.New(), XaID: "xa_chu_se", Status: "OPEN"},
		Catalog: testCatalog(),
	}
	validated := &aiextract.ValidatedProposal{
		IntentProcedureCode: "dk_khai_sinh",
		IntentConfidence:    0.2, // below ConfirmMin -> band none -> ignored
	}
	meta := &aiMeta{}
	plan, err := planTurnAI(snap, "Tôi muốn đăng ký khai sinh", validated, decision.DefaultPolicy, meta)
	if err != nil {
		t.Fatalf("planTurnAI: %v", err)
	}
	// Keyword matcher alone should still select dk_khai_sinh from the strong
	// phrase match, proving the low-confidence AI signal was ignored rather
	// than blocking the keyword fallback.
	if plan.result.ProcedureCode != "dk_khai_sinh" {
		t.Fatalf("expected keyword fallback to still pin dk_khai_sinh, got %+v", plan.result)
	}
	if meta.source != "keyword_fallback" {
		t.Fatalf("meta.source = %q, want keyword_fallback", meta.source)
	}
}

func TestPlanTurnAI_UnpinnedConfirmBand(t *testing.T) {
	snap := &repository.TurnSnapshot{
		Session: repository.Session{ID: uuid.New(), XaID: "xa_chu_se", Status: "OPEN"},
		Catalog: testCatalog(),
	}
	validated := &aiextract.ValidatedProposal{
		IntentProcedureCode: "dk_khai_sinh",
		IntentConfidence:    0.6, // within [ConfirmMin, SelectMin) -> confirm band
	}
	meta := &aiMeta{}
	plan, err := planTurnAI(snap, "một câu mơ hồ nào đó", validated, decision.DefaultPolicy, meta)
	if err != nil {
		t.Fatalf("planTurnAI: %v", err)
	}
	if plan.result.Action != decision.ActionConfirm {
		t.Fatalf("Action = %s, want %s", plan.result.Action, decision.ActionConfirm)
	}
	if meta.source != "ai" {
		t.Fatalf("meta.source = %q, want ai", meta.source)
	}
}

func TestPlanTurnAI_PinnedSwitchProposalGoesThroughConfirmNotSilentSwitch(t *testing.T) {
	catalog := testCatalog()
	snap := &repository.TurnSnapshot{
		Session:  repository.Session{ID: uuid.New(), XaID: "xa_chu_se", Status: "OPEN"},
		Catalog:  catalog,
		Pinned:   &catalog[0], // dk_khai_sinh
		PriorRaw: []byte(`{"noi_sinh":{"value":"BV Từ Dũ","status":"KNOWN"}}`),
		HadPrior: true,
	}
	validated := &aiextract.ValidatedProposal{
		IntentProcedureCode: "chung_thuc_ban_sao",
		IntentConfidence:    0.7, // confirm band
	}
	// Deliberately vague message: the keyword matcher must NOT independently
	// select chung_thuc_ban_sao here (that would exercise the pre-existing
	// keyword-switch-without-confirm path instead of the AI one under test).
	meta := &aiMeta{}
	plan, err := planTurnAI(snap, "cái kia cơ", validated, decision.DefaultPolicy, meta)
	if err != nil {
		t.Fatalf("planTurnAI: %v", err)
	}
	if plan.result.Action != decision.ActionConfirm {
		t.Fatalf("Action = %s, want %s (AI switch proposal must confirm, never silently switch)", plan.result.Action, decision.ActionConfirm)
	}
	// The pin itself must not have changed in the result (no ProcedureCode
	// pin change was persisted — PinProcedure stays false for CONFIRM_INTENT).
	if plan.save.PinProcedure {
		t.Fatalf("CONFIRM_INTENT must not pin a new procedure")
	}
}

func TestPlanTurnAI_PinnedSwitchDiscardsSlotsForTheSwitchTarget(t *testing.T) {
	catalog := testCatalog()
	snap := &repository.TurnSnapshot{
		Session:  repository.Session{ID: uuid.New(), XaID: "xa_chu_se", Status: "OPEN"},
		Catalog:  catalog,
		Pinned:   &catalog[0],
		PriorRaw: []byte(`{}`),
		HadPrior: false,
	}
	validated := &aiextract.ValidatedProposal{
		IntentProcedureCode: "chung_thuc_ban_sao",
		IntentConfidence:    0.9, // select band -> still must confirm since pinned differs
		SlotsProcedureCode:  "chung_thuc_ban_sao",
		Slots: map[string]aiextract.ValidatedSlot{
			"loai_giay_to": {Value: "CCCD", Operation: "set", Confidence: 0.9},
		},
	}
	meta := &aiMeta{}
	plan, err := planTurnAI(snap, "cái CCCD kia", validated, decision.DefaultPolicy, meta)
	if err != nil {
		t.Fatalf("planTurnAI: %v", err)
	}
	if plan.result.Action != decision.ActionConfirm {
		t.Fatalf("Action = %s, want %s", plan.result.Action, decision.ActionConfirm)
	}
	// No slot was silently applied to any procedure this turn.
	if len(plan.result.SlotState) != 0 {
		t.Fatalf("expected no slot state changes on a switch-confirm turn, got %+v", plan.result.SlotState)
	}
}

func TestPlanTurnAI_CorrectionOnlyAppliesAgainstFilledSlot(t *testing.T) {
	catalog := testCatalog()
	snap := &repository.TurnSnapshot{
		Session: repository.Session{ID: uuid.New(), XaID: "xa_chu_se", Status: "OPEN"},
		Catalog: catalog,
		Pinned:  &catalog[0],
		PriorRaw: []byte(
			`{"noi_sinh":{"value":"BV A","status":"KNOWN"},"da_ket_hon":{"value":true,"status":"KNOWN"},"co_giay_chung_sinh":{"value":true,"status":"KNOWN"}}`,
		),
		HadPrior: true,
	}
	validated := &aiextract.ValidatedProposal{
		SlotsProcedureCode: "dk_khai_sinh",
		Slots: map[string]aiextract.ValidatedSlot{
			"da_ket_hon": {Value: false, Operation: "correct", Confidence: 0.9},
		},
	}
	meta := &aiMeta{}
	plan, err := planTurnAI(snap, "à quên chưa đăng ký kết hôn", validated, decision.DefaultPolicy, meta)
	if err != nil {
		t.Fatalf("planTurnAI: %v", err)
	}
	if v := plan.result.SlotState["da_ket_hon"]; v.Value != false {
		t.Fatalf("expected da_ket_hon corrected to false, got %+v", v)
	}
	if len(meta.accepted) != 1 || meta.accepted[0] != "da_ket_hon" {
		t.Fatalf("meta.accepted = %v, want [da_ket_hon]", meta.accepted)
	}
}

func TestPlanTurnAI_NilValidatedBehavesLikePureKeyword(t *testing.T) {
	snap := &repository.TurnSnapshot{
		Session: repository.Session{ID: uuid.New(), XaID: "xa_chu_se", Status: "OPEN"},
		Catalog: testCatalog(),
	}
	meta := &aiMeta{}
	got, err := planTurnAI(snap, "hôm nay ăn gì", nil, decision.DefaultPolicy, meta)
	if err != nil {
		t.Fatalf("planTurnAI: %v", err)
	}
	if got.result.Action != decision.ActionScope {
		t.Fatalf("Action = %s, want %s", got.result.Action, decision.ActionScope)
	}
	if meta.source != "keyword_fallback" {
		t.Fatalf("meta.source = %q, want keyword_fallback", meta.source)
	}
}
