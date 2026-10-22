package drafting

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/MinhTuanDev25/citizen-assistance-system/apps/api/internal/decision"
)

// Canonical proposal identity contract (Wave 2).
//
// Material administrative claims are enumerated from the Definition itself.
// Model-invented IDs are never authoritative. Legacy Wave 1 prefixes such as
// prop_slot_<key> may be normalized only when the mapping is unambiguous.
const (
	ProposalSlot        = "slot"
	ProposalRequired    = "required"
	ProposalChecklist   = "checklist"
	ProposalConditional = "conditional"
	ProposalGuidance    = "guidance"

	ReviewPending         = "pending"
	ReviewAccepted        = "accepted"
	ReviewRejected        = "rejected"
	ReviewNeedsCorrection = "needs_correction"
	ReviewStale           = "stale"
	ReviewUnknownProposal = "unknown_proposal"
)

// MaterialProposal is one source-dependent administrative claim.
type MaterialProposal struct {
	ID          string `json:"id"`
	Type        string `json:"type"`
	Location    string `json:"location"`
	Content     string `json:"content"`
	Fingerprint string `json:"fingerprint"`
}

func SlotProposalID(key string) string       { return "slot:" + key }
func RequiredProposalID(key string) string   { return "required:" + key }
func ChecklistProposalID(i int) string       { return fmt.Sprintf("checklist:%d", i) }
func ConditionalProposalID(i int) string     { return fmt.Sprintf("conditional:%d", i) }
func GuidanceProposalID(field string) string { return "guidance:" + field }

// EnumerateMaterialProposals lists every Definition claim that requires source
// coverage before Admin approval. Pure UI/identity metadata is omitted:
// intent_examples, citations, name, procedure_code, empty guidance hints, notes.
func EnumerateMaterialProposals(raw []byte) ([]MaterialProposal, error) {
	def, err := decision.ParseDefinition(raw)
	if err != nil {
		return nil, err
	}
	out := make([]MaterialProposal, 0)

	keys := make([]string, 0, len(def.Slots))
	for key := range def.Slots {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		slot := def.Slots[key]
		content := slot.Type + "\t" + strings.TrimSpace(slot.Question)
		if len(slot.EnumValues) > 0 {
			content += "\t" + strings.Join(slot.EnumValues, ",")
		}
		out = append(out, newProposal(SlotProposalID(key), ProposalSlot, "slots."+key, content))
	}
	for _, key := range def.RequiredSlots {
		out = append(out, newProposal(RequiredProposalID(key), ProposalRequired, "required_slots."+key, key))
	}
	for i, rule := range def.Conditional {
		content := fmt.Sprintf("%s=%s require=%s", rule.WhenSlot, string(rule.Equals), strings.Join(rule.Require, ","))
		out = append(out, newProposal(ConditionalProposalID(i), ProposalConditional, fmt.Sprintf("conditional_slots[%d]", i), content))
	}
	for i, item := range def.Guidance.Checklist {
		out = append(out, newProposal(ChecklistProposalID(i), ProposalChecklist, fmt.Sprintf("guidance.checklist[%d]", i), strings.TrimSpace(item)))
	}
	addGuidance := func(field, value string) {
		value = strings.TrimSpace(value)
		if value == "" {
			return
		}
		out = append(out, newProposal(GuidanceProposalID(field), ProposalGuidance, "guidance."+field, value))
	}
	addGuidance("summary", def.Guidance.Summary)
	addGuidance("where_to_submit", def.Guidance.WhereToSubmit)
	addGuidance("fee_hint", def.Guidance.FeeHint)
	addGuidance("processing_time_hint", def.Guidance.ProcessingTimeHint)
	return out, nil
}

func newProposal(id, typ, location, content string) MaterialProposal {
	p := MaterialProposal{ID: id, Type: typ, Location: location, Content: content}
	p.Fingerprint = FingerprintProposal(p)
	return p
}

func FingerprintProposal(p MaterialProposal) string {
	canonical := p.Type + "\n" + p.ID + "\n" + p.Content
	sum := sha256.Sum256([]byte(canonical))
	return hex.EncodeToString(sum[:])
}

type IDNormalization struct {
	Original   string
	Canonical  string
	Normalized bool
	Ambiguous  bool
	Unknown    bool
}

// NormalizeProposalID maps a model-supplied ID onto the canonical contract.
// Ambiguous or colliding mappings are rejected rather than attached silently.
func NormalizeProposalID(raw string, known map[string]MaterialProposal) IDNormalization {
	raw = strings.TrimSpace(raw)
	out := IDNormalization{Original: raw, Canonical: raw}
	if raw == "" {
		out.Unknown = true
		return out
	}
	if _, ok := known[raw]; ok {
		return out
	}
	mapped, ok, ambiguous := legacyProposalID(raw)
	if ambiguous {
		out.Ambiguous = true
		out.Unknown = true
		return out
	}
	if !ok {
		out.Unknown = true
		return out
	}
	if _, exists := known[mapped]; !exists {
		out.Canonical = mapped
		out.Normalized = true
		out.Unknown = true
		return out
	}
	out.Canonical = mapped
	out.Normalized = true
	return out
}

func legacyProposalID(raw string) (string, bool, bool) {
	switch {
	case strings.HasPrefix(raw, "prop_slot_"):
		return SlotProposalID(strings.TrimPrefix(raw, "prop_slot_")), true, false
	case strings.HasPrefix(raw, "prop_required_"):
		return RequiredProposalID(strings.TrimPrefix(raw, "prop_required_")), true, false
	case strings.HasPrefix(raw, "prop_checklist_"):
		rest := strings.TrimPrefix(raw, "prop_checklist_")
		if _, err := strconv.Atoi(rest); err != nil {
			return "", false, true
		}
		return "checklist:" + rest, true, false
	case strings.HasPrefix(raw, "prop_conditional_"):
		rest := strings.TrimPrefix(raw, "prop_conditional_")
		if _, err := strconv.Atoi(rest); err != nil {
			return "", false, true
		}
		return "conditional:" + rest, true, false
	case raw == "prop_guidance_summary" || raw == "prop_summary":
		return GuidanceProposalID("summary"), true, false
	case raw == "prop_guidance_where_to_submit" || raw == "prop_where_to_submit":
		return GuidanceProposalID("where_to_submit"), true, false
	case raw == "prop_guidance_fee_hint" || raw == "prop_fee_hint":
		return GuidanceProposalID("fee_hint"), true, false
	case raw == "prop_guidance_processing_time_hint" || raw == "prop_processing_time_hint":
		return GuidanceProposalID("processing_time_hint"), true, false
	case strings.HasPrefix(raw, "prop_"):
		return "", false, true
	default:
		return "", false, false
	}
}

func proposalsByID(items []MaterialProposal) map[string]MaterialProposal {
	out := make(map[string]MaterialProposal, len(items))
	for _, item := range items {
		out[item.ID] = item
	}
	return out
}
