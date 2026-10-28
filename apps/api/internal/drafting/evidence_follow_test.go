package drafting

import (
	"encoding/json"
	"testing"
)

func checklistDef(items ...string) json.RawMessage {
	raw, _ := json.Marshal(map[string]any{"procedure_code": "p", "name": "n", "slots": map[string]any{},
		"guidance": map[string]any{"summary": "", "checklist": items}})
	return raw
}

func boundEvidence(def json.RawMessage, ids ...string) []EvidenceItem {
	props, _ := EnumerateMaterialProposals(def)
	byID := proposalsByID(props)
	out := []EvidenceItem{}
	for _, id := range ids {
		out = append(out, EvidenceItem{ProposalID: id, RawProposalID: id, BoundFingerprint: byID[id].Fingerprint, ReviewStatus: ReviewAccepted, Quotation: id})
	}
	return out
}

// Removing item B from [A, B, C]: B's evidence goes, C's evidence moves to
// checklist:1 and stays bound (not stale).
func TestEvidenceFollowsRemovedListItem(t *testing.T) {
	before := checklistDef("A", "B", "C")
	after := checklistDef("A", "C")
	got := invalidateStaleEvidence(followEditedProposals(boundEvidence(before, "checklist:0", "checklist:1", "checklist:2"), before, after), after)
	if len(got) != 2 || got[0].ProposalID != "checklist:0" || got[1].ProposalID != "checklist:1" || got[1].Quotation != "checklist:2" {
		t.Fatalf("evidence = %+v", got)
	}
	for _, e := range got {
		if e.ReviewStatus != ReviewAccepted {
			t.Fatalf("moved evidence must stay accepted: %+v", e)
		}
	}
}

// Removing the last items (the real case: #6..#10 rejected one by one).
func TestEvidenceOfRemovedTailIsDropped(t *testing.T) {
	before := checklistDef("A", "B", "C")
	after := checklistDef("A")
	got := followEditedProposals(boundEvidence(before, "checklist:0", "checklist:1", "checklist:2"), before, after)
	if len(got) != 1 || got[0].ProposalID != "checklist:0" {
		t.Fatalf("evidence = %+v", got)
	}
}

// An item edited in place keeps its evidence, which then turns stale.
func TestEditedItemEvidenceTurnsStale(t *testing.T) {
	before := checklistDef("A", "B")
	after := checklistDef("A", "B sửa")
	got := invalidateStaleEvidence(followEditedProposals(boundEvidence(before, "checklist:0", "checklist:1"), before, after), after)
	if len(got) != 2 || got[1].ReviewStatus != ReviewStale {
		t.Fatalf("evidence = %+v", got)
	}
}

// Evidence already pointing at nothing (left by an earlier removal) is cleaned
// up by the next save.
func TestOrphanEvidenceIsDroppedOnSave(t *testing.T) {
	def := checklistDef("A")
	items := append(boundEvidence(def, "checklist:0"), EvidenceItem{ProposalID: "checklist:5", ReviewStatus: ReviewUnknownProposal})
	got := followEditedProposals(items, def, def)
	if len(got) != 1 || got[0].ProposalID != "checklist:0" {
		t.Fatalf("evidence = %+v", got)
	}
}
