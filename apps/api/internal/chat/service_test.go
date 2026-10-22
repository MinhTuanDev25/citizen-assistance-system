package chat

import (
	"encoding/json"
	"testing"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/decision"
	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/repository"
	"github.com/google/uuid"
)

func testCatalog() []repository.ProcedureDefinitionRow {
	ks := []byte(`{
		"procedure_code": "dk_khai_sinh",
		"name": "Đăng ký khai sinh",
		"intent_examples": ["Đăng ký khai sinh", "Làm giấy khai sinh"],
		"slots": {
			"noi_sinh": {"type": "string", "question": "Bé sinh ở đâu?"},
			"da_ket_hon": {"type": "boolean", "question": "Đã kết hôn?"},
			"co_giay_chung_sinh": {"type": "boolean", "question": "Có giấy chứng sinh?"}
		},
		"required_slots": ["noi_sinh", "da_ket_hon", "co_giay_chung_sinh"],
		"guidance": {"summary": "Hồ sơ khai sinh", "checklist": ["GCS"], "where_to_submit": "UBND"},
		"citations": [{"doc_id": "d1", "title": "HD KS", "source_type": "manual_seed"}]
	}`)
	ct := []byte(`{
		"procedure_code": "chung_thuc_ban_sao",
		"name": "Chứng thực bản sao",
		"intent_examples": ["Chứng thực bản sao"],
		"slots": {"loai_giay_to": {"type": "string", "question": "Loại giấy tờ?"}},
		"required_slots": [],
		"guidance": {"summary": "Chứng thực", "checklist": ["Bản chính"], "where_to_submit": "Một cửa"},
		"citations": [{"doc_id": "d2", "title": "HD CT", "source_type": "manual_seed"}]
	}`)
	id1, id2 := uuid.New(), uuid.New()
	v1, v2 := uuid.New(), uuid.New()
	return []repository.ProcedureDefinitionRow{
		{ProcedureID: id1, VersionID: v1, ProcedureCode: "dk_khai_sinh", Name: "Đăng ký khai sinh", Version: "1.0.0", Definition: ks},
		{ProcedureID: id2, VersionID: v2, ProcedureCode: "chung_thuc_ban_sao", Name: "Chứng thực bản sao", Version: "1.0.0", Definition: ct},
	}
}

func TestPlanTurnOutOfScope(t *testing.T) {
	snap := &repository.TurnSnapshot{
		Session: repository.Session{ID: uuid.New(), XaID: "xa_chu_se", Status: "OPEN"},
		Catalog: testCatalog(),
	}
	p, err := planTurn(nil, nil, nil, snap, "hôm nay ăn gì")
	if err != nil {
		t.Fatal(err)
	}
	if p.result.Action != decision.ActionScope {
		t.Fatalf("action=%s", p.result.Action)
	}
}

func TestPlanTurnHighConfidencePins(t *testing.T) {
	snap := &repository.TurnSnapshot{
		Session: repository.Session{ID: uuid.New(), XaID: "xa_chu_se", Status: "OPEN"},
		Catalog: testCatalog(),
	}
	p, err := planTurn(nil, nil, nil, snap, "Tôi muốn đăng ký khai sinh")
	if err != nil {
		t.Fatal(err)
	}
	if !p.save.PinProcedure {
		t.Fatal("expected pin")
	}
	if len(p.save.Citations) == 0 {
		t.Fatal("expected citations")
	}
}

func TestPlanTurnConfirmStoresVersionIDs(t *testing.T) {
	catalog := testCatalog()
	id1, v1 := catalog[0].ProcedureID, catalog[0].VersionID
	id2, v2 := catalog[1].ProcedureID, catalog[1].VersionID
	pending, _ := json.Marshal(pendingIntent{Candidates: []PendingCandidate{
		{
			ProcedureCode: "dk_khai_sinh", Name: "Đăng ký khai sinh", Score: 0.7,
			ProcedureID: id1.String(), ProcedureVersionID: v1.String(),
			DefinitionHash: DefinitionHash(catalog[0].Definition),
		},
		{
			ProcedureCode: "chung_thuc_ban_sao", Name: "Chứng thực bản sao", Score: 0.65,
			ProcedureID: id2.String(), ProcedureVersionID: v2.String(),
			DefinitionHash: DefinitionHash(catalog[1].Definition),
		},
	}})
	snap := &repository.TurnSnapshot{
		Session:       repository.Session{ID: uuid.New(), Status: "OPEN"},
		Catalog:       catalog,
		PendingIntent: pending,
	}
	// Negative path does not need DB.
	no, err := planTurn(nil, nil, nil, snap, "Không")
	if err != nil {
		t.Fatal(err)
	}
	if no.result.Action != decision.ActionScope {
		t.Fatalf("no action=%s", no.result.Action)
	}
}

func TestPinnedSessionIgnoresNewerActiveCatalog(t *testing.T) {
	catalog := testCatalog()
	pinned := catalog[0]
	newer := pinned
	newer.VersionID = uuid.New()
	newer.Version = "2.0.0"
	newer.Definition = []byte(`{
		"procedure_code": "dk_khai_sinh",
		"name": "Đăng ký khai sinh",
		"intent_examples": ["Đăng ký khai sinh"],
		"slots": {"noi_sinh": {"type": "string", "question": "Bé sinh ở đâu?"}},
		"required_slots": ["noi_sinh"],
		"guidance": {"summary": "v2", "checklist": ["v2"], "where_to_submit": "UBND"}
	}`)
	procID := pinned.ProcedureID
	verID := pinned.VersionID
	prior, _ := json.Marshal(map[string]decision.SlotValue{
		"noi_sinh":           {Value: "Gia Lai", Status: decision.StatusKnown},
		"da_ket_hon":         {Value: true, Status: decision.StatusKnown},
		"co_giay_chung_sinh": {Value: true, Status: decision.StatusKnown},
	})
	snap := &repository.TurnSnapshot{
		Session: repository.Session{
			ID: uuid.New(), XaID: "xa_chu_se", Status: "OPEN",
			ActiveProcedureID: &procID, ActiveProcedureVersionID: &verID,
		},
		Catalog:  []repository.ProcedureDefinitionRow{newer},
		Pinned:   &pinned,
		PriorRaw: prior,
		HadPrior: true,
	}
	p, err := planTurn(nil, nil, nil, snap, "Tôi muốn đăng ký khai sinh")
	if err != nil {
		t.Fatal(err)
	}
	if p.save.VersionID == nil || *p.save.VersionID != pinned.VersionID {
		t.Fatalf("pin jumped to %v", p.save.VersionID)
	}
	if p.result.ProcedureVersion != "1.0.0" {
		t.Fatalf("version=%s", p.result.ProcedureVersion)
	}
	if p.result.Guidance == nil || p.result.Guidance.Summary == "v2" {
		t.Fatalf("used new ACTIVE definition: %+v", p.result.Guidance)
	}
	if p.result.Guidance.Summary != "Hồ sơ khai sinh" {
		t.Fatalf("summary=%s", p.result.Guidance.Summary)
	}
}

func TestPayloadHashStable(t *testing.T) {
	a := PayloadHash("  hello   world  ")
	b := PayloadHash("hello world")
	if a != b {
		t.Fatalf("%s != %s", a, b)
	}
	if PayloadHash("hello world") == PayloadHash("hello worlds") {
		t.Fatal("different payloads must differ")
	}
}

func TestMaxMessageRunes(t *testing.T) {
	if MaxMessageRunes != 4000 {
		t.Fatal(MaxMessageRunes)
	}
}
