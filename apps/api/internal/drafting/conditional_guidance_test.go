package drafting

import (
	"encoding/json"
	"strings"
	"testing"
)

// Batch 1 asks truong_hop_tre; batch 3 (another page range) lists the
// abandoned-child paper for it. The merge keeps the item with its condition,
// dedupes a repeat, keeps evidence on conditional_checklist:N, and drops an
// item whose question exists nowhere.
func TestAggregateConditionalGuidance(t *testing.T) {
	mk := func(label string, slots map[string]any, cond, notes []any, evidence []EvidenceItem) Contribution {
		def, _ := json.Marshal(map[string]any{"procedure_code": "khai_sinh_v1", "name": "Đăng ký khai sinh", "slots": slots,
			"guidance": map[string]any{"summary": "", "checklist": []string{}, "conditional_checklist": cond, "conditional_notes": notes}})
		return Contribution{Label: label, PageStart: 9, PageEnd: 21, Definition: def, Evidence: evidence}
	}
	rule := func(slot string, equals any, item string) map[string]any {
		return map[string]any{"when": map[string]any{"slot": slot, "equals": equals}, "item": item}
	}
	slots := map[string]any{"truong_hop_tre": map[string]any{"type": "enum", "question": "Trường hợp?", "enum_values": []string{"tre_sinh_thuong", "tre_bi_bo_roi"}}}
	ev := EvidenceItem{ProposalID: "conditional_checklist:0", Kind: "checklist", Quotation: "biên bản"}
	agg, err := Aggregate("khai_sinh_v1", "Đăng ký khai sinh", []Contribution{
		mk("batch:1", slots, []any{rule("truong_hop_tre", "tre_sinh_thuong", "Giấy chứng sinh")}, nil, nil),
		mk("batch:3", map[string]any{}, []any{
			rule("truong_hop_tre", "tre_bi_bo_roi", "Biên bản về việc trẻ bị bỏ rơi"),
			rule("truong_hop_tre", "tre_sinh_thuong", "giấy chứng  sinh"),
			rule("hinh_thuc_nop", "truc_tuyen", "Mẫu hộ tịch điện tử"),
		}, []any{rule("truong_hop_tre", "tre_bi_bo_roi", "Niêm yết 07 ngày")}, []EvidenceItem{ev}),
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	var def aggDef
	_ = json.Unmarshal(agg.Definition, &def)
	got := def.Guidance.ConditionalChecklist
	if len(got) != 2 || got[0].Item != "Giấy chứng sinh" || got[1].Item != "Biên bản về việc trẻ bị bỏ rơi" || got[1].When.Slot != "truong_hop_tre" {
		t.Fatalf("conditional checklist = %+v", got)
	}
	if len(def.Guidance.ConditionalNotes) != 1 {
		t.Fatalf("conditional notes = %+v", def.Guidance.ConditionalNotes)
	}
	dropped := false
	for _, w := range agg.Warnings {
		dropped = dropped || strings.Contains(w, "Mẫu hộ tịch điện tử")
	}
	if !dropped {
		t.Fatalf("an item for a question that does not exist must be reported: %v", agg.Warnings)
	}
	props, _ := EnumerateMaterialProposals(agg.Definition)
	ids := map[string]string{}
	for _, p := range props {
		ids[p.ID] = p.Content
	}
	if ids["conditional_checklist:1"] != `truong_hop_tre="tre_bi_bo_roi"`+"\tBiên bản về việc trẻ bị bỏ rơi" {
		t.Fatalf("proposals = %+v", ids)
	}
	// batch 3's conditional_checklist:0 became merged conditional_checklist:1.
	found := false
	for _, p := range agg.Provenance {
		if p.ProposalID == "conditional_checklist:1" {
			found = true
		}
	}
	if !found {
		t.Fatalf("provenance must follow the merged index: %+v", agg.Provenance)
	}
}
